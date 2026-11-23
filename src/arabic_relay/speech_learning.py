"""Opt-in: saves dictation audio with its words in the local training inbox (scripts/retrain_when_idle.py).

Off unless the user registered an inbox, which writes speech-learning.json. Each finished transcript gets an ID, and the
page that made the recording sends its audio once under that ID. The last few recordings stay in memory for half an hour,
so History can correct the last one after the shortcut window closed. On disk:
- `.pending` holds machine text until a second local model agrees with it while the computer is idle (automatic mode only);
- the inbox itself gets a recording when the user saves a correction, which replaces a machine label.
Nothing is uploaded, and nothing here runs before the transcript is ready.
"""
from collections import OrderedDict
import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
import threading
import time
import unicodedata

from . import models, paths

MAX_SECONDS = 30.0                   # the trainer's limit; a longer recording is never saved
MAX_AUDIO = 8 * 1024 * 1024          # far above 30 seconds of the page's 64 kbit/s recordings
MAX_TEXT = 16 * 1024
MAX_PENDING = 500                    # recordings waiting for the second model
MIN_FREE_BYTES = 2 * 10 ** 9
KEPT, KEPT_SECONDS = 3, 1800         # recordings in memory, and for how long
STALE_SECONDS = 24 * 3600
KINDS = {'audio/webm': ('.webm', lambda b: b.startswith(b'\x1aE\xdf\xa3')),
         'audio/mp4': ('.mp4', lambda b: b[4:8] == b'ftyp')}
EXPIRED = 'التسجيل ده مش متاح للتصحيح دلوقتي.'
NOT_SAVED = 'ما قدرتش أحفظ التصحيح.'
NO_WORDS = 'اكتب الكلام اللي اتقال.'

LOCK = threading.Lock()              # RECORDINGS and LAST; a transcript waits for it, so no file work under it
FILES = threading.Lock()             # the inbox, taken before LOCK
RECORDINGS = OrderedDict()           # id -> one recognized recording
LAST = None                          # the id History may correct
LATEST = None                        # latest recognition, including one that cannot be kept


# --- Settings --------------------------------------------------------------------------------------------------

def absolute(value):
    return Path(value) if isinstance(value, str) and value.strip() and '\0' not in value and Path(value).is_absolute() else None


def on_network(path):
    """True for a share or a mapped network drive: recordings must not leave this computer."""
    if os.name != 'nt':
        return False
    return str(path).startswith(('\\\\', '//')) or ctypes.windll.kernel32.GetDriveTypeW(path.drive + '\\') == 4


def is_link(path):
    return path.is_symlink() or path.is_junction()


def configured():
    """The saved choice, or None. Any doubt means off."""
    try:
        source = paths.DATA / 'speech-learning.json'
        if source.stat().st_size > 64 * 1024:
            return None
        saved = json.loads(source.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(saved, dict) or type(saved.get('automaticLabels')) is not bool or absolute(saved.get('inbox')) is None:
        return None
    return saved


def settings():
    """{'inbox', 'automatic', 'verify'} when capture is on and its folder is usable on this computer, otherwise None."""
    saved = configured()
    if saved is None:
        return None
    try:
        inbox = absolute(saved['inbox'])
        if on_network(inbox):
            return None
        inbox = inbox.resolve(strict=True)
        if on_network(inbox) or not inbox.is_dir():
            return None
    except (OSError, ValueError):
        return None
    return {'inbox': inbox, 'automatic': saved['automaticLabels'], 'verify': absolute(saved.get('verifyModel'))}


def mode():
    current = settings()
    return None if current is None else 'automatic' if current['automatic'] else 'corrections'


def verifier_for(model, verify):
    """The resolved second model, only when it surely is not the one that wrote the words: agreeing with itself proves
    nothing. None for the same or an unknown model."""
    try:
        model, verify = Path(model).resolve(strict=True), Path(verify).resolve(strict=True)
        labels = {models.model_label(model), models.model_label(verify)}
        if (len(labels) < 2 or 'Whisper (your model)' in labels
                # A copy under another name still has the same weights.
                or (model / 'model.bin').stat().st_size == (verify / 'model.bin').stat().st_size):
            return None
        return verify
    except (OSError, TypeError, ValueError):
        return None


# --- Recordings in memory --------------------------------------------------------------------------------------

def clean(text):
    """One line of plain words, or None."""
    if not isinstance(text, str):
        return None
    text = ' '.join(text.split())
    if not any(c.isalnum() for c in text) or any(unicodedata.category(c) in ('Cc', 'Cs') for c in text):
        return None
    return text if len(text.encode()) <= MAX_TEXT else None


def forget_expired():
    global LAST
    now = time.monotonic()
    with LOCK:
        for key in [key for key, entry in RECORDINGS.items() if now - entry['created'] >= KEPT_SECONDS]:
            del RECORDINGS[key]
        if LAST not in RECORDINGS:
            LAST = None


def forget_all():
    global LAST, LATEST
    with LOCK:
        for entry in RECORDINGS.values():
            if timer := entry.get('timer'):
                timer.cancel()
        RECORDINGS.clear()
        LAST = None
        LATEST = None


def find(recording):
    forget_expired()
    with LOCK:
        entry = RECORDINGS.get(recording) if isinstance(recording, str) else None
    if entry is None:
        raise ValueError(EXPIRED)
    return entry


def offer(response, model=None, seconds=None, audio=None):
    """Adds `learning` to a finished transcript whose recording may be saved. Never fails or delays the transcript.
    `model` is the folder that decoded it; without one (an online service) the recording is not for learning."""
    global LAST, LATEST
    try:
        recording = secrets.token_urlsafe(24)
        with LOCK:
            LATEST = recording
            LAST = None                  # History corrects the last dictation, never an older one by mistake
        heard = clean(response.get('rawText'))
        if (model is None or heard is None or configured() is None or (audio is not None and len(audio) > MAX_AUDIO)
                or (seconds is not None and not 0.2 <= seconds <= MAX_SECONDS)):
            return response
        forget_expired()
        with LOCK:
            while len(RECORDINGS) >= KEPT:
                _, evicted = RECORDINGS.popitem(last=False)
                if timer := evicted.get('timer'):
                    timer.cancel()
            # A whole upload already names its audio; a live recording is named by the first audio sent for it.
            RECORDINGS[recording] = {'heard': heard, 'model': Path(model), 'created': time.monotonic(), 'at': time.time(),
                                     'digest': hashlib.sha256(audio).hexdigest() if audio is not None else None,
                                     'audio': None, 'suffix': None, 'corrected': None}
        response['learning'] = {'id': recording}
    except Exception:
        logging.exception('Speech learning could not follow a recording')
    return response


# --- Files -----------------------------------------------------------------------------------------------------

def write(path, data):
    """The whole file under its final name, or nothing. Replaces a name; never edits a file another name shares."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix='.nuvra-', suffix='.tmp')
    try:
        with os.fdopen(handle, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:      # a scanner or the trainer can hold the file briefly on Windows
                if attempt == 5:
                    raise
                time.sleep(0.05 * 2 ** attempt)
    finally:
        Path(temporary).unlink(missing_ok=True)


def waiting_room(inbox, create=False):
    folder = inbox / '.pending'
    if create:
        folder.mkdir(exist_ok=True)
    return folder if folder.is_dir() and not is_link(folder) else None


def tidy(folder):
    """Counts what waits, and clears what a stopped app or an interrupted check left behind: temporary files and
    incomplete sets. Finished sets are the verifier's."""
    now, files, stems = time.time(), list(folder.iterdir()), {}
    for path in files:
        stems.setdefault(path.stem, set()).add(path.suffix)
    for path in files:
        whole = '.txt' in stems[path.stem] and stems[path.stem] & {'.webm', '.mp4'}
        if re.fullmatch(r'\.nuvra-.*\.tmp', path.name) or (not whole and re.fullmatch('nuvra-[0-9a-f]{64}', path.stem)):
            try:
                if now - path.lstat().st_mtime > STALE_SECONDS:
                    path.unlink()
            except OSError:
                pass
    return sum(path.suffix == '.txt' for path in files)


# --- Saving ----------------------------------------------------------------------------------------------------

def usable():
    """Settings as they are now: turning capture off also releases what is in memory."""
    current = settings()
    if current is None:
        forget_all()
    return current


def keep(recording, audio, content_type):
    """The page accepted this transcript and sends its recording: kept in memory, and staged in automatic mode."""
    global LAST
    kind = (content_type or '').split(';', 1)[0].strip().lower()
    if kind not in KINDS or not 0 < len(audio) <= MAX_AUDIO or not KINDS[kind][1](audio):
        raise ValueError(EXPIRED)
    suffix, digest = KINDS[kind][0], hashlib.sha256(audio).hexdigest()
    entry = find(recording)
    with FILES:
        current = usable()
        if current is None:
            return {'kept': False, 'staged': False}
        with LOCK:
            if entry['digest'] not in (None, digest):
                raise ValueError(EXPIRED)        # other audio can never take this recording's words
            entry.update(digest=digest, audio=audio, suffix=suffix)
            # A late upload of an earlier recording must not become the last one again.
            if recording in RECORDINGS and recording == LATEST:
                LAST = recording
            if recording in RECORDINGS and 'timer' not in entry:
                remaining = max(0, entry['created'] + KEPT_SECONDS - time.monotonic())
                timer = threading.Timer(remaining + 1, forget_expired)
                timer.daemon = True
                entry['timer'] = timer
                timer.start()
        try:
            staged = stage(entry, current)
        except OSError:
            logging.exception('A recording could not wait for the second model')
            staged = False
    return {'kept': True, 'staged': staged}


def stage(entry, current):
    inbox, stem = current['inbox'], 'nuvra-' + entry['digest']
    verify = verifier_for(entry['model'], current['verify']) if current['automatic'] else None
    if (verify is None or entry['corrected'] is not None or (inbox / (stem + '.txt')).exists()
            or shutil.disk_usage(inbox).free < MIN_FREE_BYTES):
        return False
    folder = waiting_room(inbox, create=True)
    if folder is None or tidy(folder) >= MAX_PENDING or (folder / (stem + '.txt')).exists():
        return False
    write(folder / (stem + entry['suffix']), entry['audio'])
    # The trainer drops this set if another second model is registered later: it could be the one that wrote the words.
    write(folder / (stem + '.verifier'), str(verify).encode())
    write(folder / (stem + '.txt'), entry['heard'].encode())     # last: the verifier starts from the text
    return True


def last():
    """What History offers to correct: the last dictation's own words, never the pasted result."""
    if usable() is None:
        return {'id': None}
    forget_expired()
    with LOCK:
        entry = RECORDINGS.get(LAST)
        if entry is None or entry['audio'] is None:
            return {'id': None}
        return {'id': LAST, 'text': entry['corrected'] or entry['heard'], 'at': entry['at'],
                'corrected': entry['corrected'] is not None}


def correct(recording, text):
    """Words the user saved in History. They may rewrite everything: recognition can be wrong throughout."""
    text = clean(text)
    if text is None:
        raise ValueError(NO_WORDS)
    entry = find(recording)
    with FILES:
        current = usable()
        if current is None or entry['audio'] is None:
            raise ValueError(EXPIRED)
        if text == (entry['corrected'] or entry['heard']):
            return {'saved': False}
        inbox, stem = current['inbox'], 'nuvra-' + entry['digest']
        try:
            if shutil.disk_usage(inbox).free < MIN_FREE_BYTES:
                raise ValueError(NOT_SAVED)
            write(inbox / (stem + entry['suffix']), entry['audio'])
            write(inbox / (stem + '.txt'), text.encode())        # replaces a machine label, also one linked meanwhile
        except OSError:
            logging.exception('A correction could not be saved')
            raise ValueError(NOT_SAVED) from None
        entry['corrected'] = text
        folder = waiting_room(inbox)
        if folder is not None:
            for suffix in ('.txt', '.verifier', entry['suffix']):
                try:
                    (folder / (stem + suffix)).unlink(missing_ok=True)
                except OSError:          # the verifier is reading it; the label saved above still wins
                    logging.info('Speech learning: a waiting file is in use')
    return {'saved': True}
