"""Retrains the speech model when you have added corrected recordings and the computer is idle (Windows).

    python scripts/retrain_when_idle.py register      # check every 15 minutes, for this Windows account only
    python scripts/retrain_when_idle.py status        # what it is doing, what it ignored and why
    python scripts/retrain_when_idle.py unregister    # stop checking; recordings and models stay
    python scripts/retrain_when_idle.py run [--now]   # one check, as scheduled (--now: do not wait for idle)

Put each recording in the inbox folder next to its corrected transcript: meeting.wav (or .webm, .mp4) and meeting.txt,
UTF-8. Both files are needed; nothing is recorded, transcribed or collected here, and nothing leaves the computer.
Recordings without a transcript are ignored. Idle time alone never trains: only a new recording or a changed
transcript does, together with the public corpus, and only with nobody using the computer, on mains power, with the
GPU free. The run stops as soon as you come back and is tried again later.

A new model must pass the same dev gates as train_speech.py. It is written to its own folder and never activated:
the model chosen in Nuvra is not read or changed.
"""
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_speech  # noqa: E402
from arabic_relay import paths  # noqa: E402

TASK = 'Nuvra Idle Retraining'
RUN_KEY = r'Software\Microsoft\Windows\CurrentVersion\Run'
AUDIO = ('.wav', '.webm', '.mp4')
EVERY_MINUTES, IDLE_MINUTES = 15, 10
SETTLE_SECONDS = 120                 # files still being copied or edited are left for the next check
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_TEXT_BYTES = 16 * 1024
GPU_FREE_MB, GPU_BUSY_PERCENT, DISK_FREE_BYTES = 2500, 30, 8 * 10 ** 9
TRAINING = ['--batch-size', '8', '--eval-batch-size', '8', '--epochs', '4']
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
ACCESS_DENIED, NOT_FOUND = -2147024891, -2147024894   # HRESULTs
INTERRUPTED, GAVE_WAY = 'interrupted', 75


# --- Inbox -----------------------------------------------------------------------------------------------------

def audio_seconds(path):
    from faster_whisper.audio import decode_audio
    return len(decode_audio(str(path), sampling_rate=train_speech.RATE)) / train_speech.RATE


def listing(folder):
    """Regular files directly inside the inbox, with what tells a change apart."""
    files = {}
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in (*AUDIO, '.txt') and path.is_file() and not path.is_symlink():
                info = path.stat()
                files[path] = (info.st_size, info.st_mtime_ns, max(info.st_mtime, info.st_ctime))
    return files


def signature(files, manifest):
    try:
        info = manifest.stat()
        public = [info.st_size, info.st_mtime_ns]
    except OSError:
        public = None
    rows = [[path.name, size, modified] for path, (size, modified, _) in files.items()]
    return hashlib.sha256(json.dumps([rows, public]).encode()).hexdigest()


def held_out(manifest):
    """(audio hashes, sentences, ids) of the public corpus that an inbox recording must not repeat."""
    try:
        clips = json.loads(manifest.read_text(encoding='utf-8-sig'))['clips']
        hashes, sentences = set(), set()
        for clip in clips:
            if clip['split'] == 'train':
                continue
            path = manifest.parent / clip['path']
            hashes.add(train_speech.sha256_file(path))
            sentences.add(' '.join(train_speech.words(clip['reference'])))
        return hashes, sentences, {clip['id'] for clip in clips}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise train_speech.InputError(f'cannot read {manifest}: {error}') from None


def transcript(path):
    if path.stat().st_size > MAX_TEXT_BYTES:
        return None, 'the transcript is too long'
    try:
        text = ' '.join(path.read_bytes().decode('utf-8-sig').split())
    except UnicodeDecodeError:
        return None, 'the transcript is not UTF-8'
    return (text, None) if train_speech.words(text) else (None, 'the transcript is empty')


def agreement_words(text):
    """Ignore casing, Arabic vocalization and sentence punctuation, keeping numbers and technical names intact."""
    text = unicodedata.normalize('NFC', text).casefold()
    text = ''.join(c for c in text if not '\u064b' <= c <= '\u0652')
    return [w for word in text.split() if (w := word.rstrip('.,!?،؟؛'))]


def load_verifier(folder):
    from faster_whisper import WhisperModel
    return WhisperModel(str(folder), device='cpu', compute_type='int8', cpu_threads=4, local_files_only=True)


def verify_transcript(model, audio):
    segments, _ = model.transcribe(str(audio), language='ar', task='transcribe', beam_size=5,
                                   condition_on_previous_text=False, vad_filter=True)
    return ' '.join(segment.text.strip() for segment in segments)


def publish_agreement(inbox, audio, label):
    """Link complete files without replacing an existing pair or an intervening human correction."""
    target = inbox / audio.name
    try:
        os.link(audio, target)
    except FileExistsError:
        if target.is_symlink() or not target.is_file():
            return False
        if not target.samefile(audio) and train_speech.sha256_file(target) != train_speech.sha256_file(audio):
            return False
    try:
        os.link(label, inbox / label.name)
        return True
    except FileExistsError:
        return False


def verify_pending(inbox, model_folder):
    """The app publishes the proposal last. Never overwrite a user's correction, even during verification."""
    folder = inbox / '.pending'
    if folder.resolve().parent != inbox.resolve() or folder.is_symlink():
        raise train_speech.InputError('pending recordings must be stored directly in the inbox')
    model = None
    matched = skipped = 0
    for label in sorted(folder.glob('*.txt')):
        if label.is_symlink() or not label.is_file():
            continue
        sounds = [p for ext in AUDIO if (p := label.with_suffix(ext)).is_file() and not p.is_symlink()]
        if len(sounds) != 1:
            continue
        audio, = sounds
        try:
            proposed, _ = transcript(label)
            eligible = proposed and not (inbox / label.name).exists() and 0 < audio.stat().st_size <= MAX_AUDIO_BYTES
            binding = label.with_suffix('.verifier')
            if eligible and label.stem.startswith('nuvra-'):
                try:
                    eligible = (binding.is_file() and not binding.is_symlink() and binding.stat().st_size <= 8192
                                and Path(binding.read_text(encoding='utf-8').strip()) == model_folder)
                except (UnicodeError, FileNotFoundError):
                    eligible = False
            if eligible:
                try:
                    eligible = 0.2 <= audio_seconds(audio) <= train_speech.MAX_SECONDS
                except (ValueError, RuntimeError, OSError):
                    eligible = False
            if eligible:
                if model is None:
                    model = load_verifier(model_folder)
                words = agreement_words(proposed)
                if words and words == agreement_words(verify_transcript(model, audio)):
                    matched += int(publish_agreement(inbox, audio, label))
                else:
                    skipped += 1
            label.unlink(missing_ok=True)
            audio.unlink(missing_ok=True)
            binding.unlink(missing_ok=True)
        except FileNotFoundError:
            continue  # a foreground correction can supersede this pending pair
    return matched, skipped


def scan(folder, manifest, durations, now=None):
    """Usable pairs as manifest clips, the ignored files with reasons, and whether some files are still settling.

    `durations` remembers decoded lengths by audio hash, so an unchanged recording is decoded only once.
    """
    now = time.time() if now is None else now
    files = listing(folder)
    ignored, settling, found, conflicting = [], False, {}, set()
    hashes, sentences, ids = held_out(manifest)
    stems = {}
    for path in files:
        stems.setdefault(path.stem.casefold(), []).append(path)

    def skip(path, why):
        ignored.append({'file': path.name, 'why': why})

    for group in stems.values():
        sounds = [p for p in group if p.suffix.lower() in AUDIO]
        texts = [p for p in group if p.suffix.lower() == '.txt']
        if not sounds or not texts:
            skip(group[0], 'no transcript yet' if sounds else 'no recording with this name')
            continue
        if len(sounds) > 1 or len(texts) > 1:
            skip(group[0], 'several files share this name')
            continue
        sound, text_file = sounds[0], texts[0]
        if any(abs(now - files[p][2]) < SETTLE_SECONDS for p in group):
            settling = True
            skip(sound, 'still being added')
            continue
        try:
            text, why = transcript(text_file)
            if why:
                skip(text_file, why)
                continue
            if not 0 < files[sound][0] <= MAX_AUDIO_BYTES:
                skip(sound, 'the recording is empty or too large')
                continue
            digest = train_speech.sha256_file(sound)
        except OSError:                                   # another program still holds the file
            settling = True
            skip(sound, 'the file is in use')
            continue
        if digest in conflicting:
            skip(sound, 'the same recording has different transcripts')
            continue
        if digest in found:
            if found[digest]['reference'] != text:
                conflicting.add(digest)
                skip(folder / found.pop(digest)['path'], 'the same recording has different transcripts')
                skip(sound, 'the same recording has different transcripts')
            else:
                skip(sound, 'the same recording is already in the inbox')
            continue
        sentence = train_speech.words(text)
        if digest in hashes or (len(sentence) >= 3 and ' '.join(sentence) in sentences):
            skip(sound, 'reserved for evaluation (it repeats a dev or test clip)')
            continue
        if digest not in durations:
            try:
                durations[digest] = audio_seconds(sound)
            except Exception as error:                    # undecodable input of any kind
                durations[digest] = None
                print(f'{sound.name}: {type(error).__name__}', file=sys.stderr)
        if durations[digest] is None:
            skip(sound, 'the recording cannot be decoded')
            continue
        if not 0.2 <= durations[digest] <= train_speech.MAX_SECONDS:
            skip(sound, f'{durations[digest]:.1f} seconds long (the limit is 30)')
            continue
        clip_id = 'inbox-' + digest[:24]                  # from the audio: renaming is no change, a new label is
        if clip_id in ids:
            skip(sound, 'already in the corpus')
            continue
        found[digest] = {'id': clip_id, 'split': 'train', 'group': 'personal', 'path': sound.name,
                         'reference': text, 'audio_sha256': digest}
    return sorted(found.values(), key=lambda clip: clip['id']), ignored, settling


# --- Conditions ------------------------------------------------------------------------------------------------

def idle_seconds():
    class LastInput(ctypes.Structure):
        _fields_ = [('cbSize', ctypes.c_uint), ('dwTime', ctypes.c_uint)]
    info = LastInput(cbSize=ctypes.sizeof(LastInput))
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0                                        # unknown counts as in use
    return ((ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000


def on_battery():
    """True unless Windows confirms mains power: an unknown state is not mains."""
    class Power(ctypes.Structure):
        _fields_ = [('ac', ctypes.c_ubyte), ('flag', ctypes.c_ubyte), ('percent', ctypes.c_ubyte),
                    ('saver', ctypes.c_ubyte), ('life', ctypes.c_uint), ('full', ctypes.c_uint)]
    power = Power()
    return not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)) or power.ac != 1


def gpu_state():
    """(free MB, busy percent) of the first NVIDIA GPU, or None."""
    try:
        done = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        free, busy = done.stdout.splitlines()[0].split(',')
        return int(free), int(busy)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def blocked(root, idle_minutes):
    """Why training must wait, or None."""
    if idle_seconds() < idle_minutes * 60:
        return 'waiting until the computer is idle'
    if on_battery():
        return 'waiting for mains power'
    gpu = gpu_state()
    if gpu is None:
        return 'waiting for an NVIDIA GPU'
    if gpu[0] < GPU_FREE_MB or gpu[1] >= GPU_BUSY_PERCENT:
        return 'waiting until the GPU is free'
    if shutil.disk_usage(root).free < DISK_FREE_BYTES:
        return 'waiting for 8 GB of free disk space'
    return None


class Watchdog:
    """Ends this process when somebody uses the computer again: that frees the GPU and the training lock at once.

    Every file the training writes appears under its final name only when complete, so stopping is always safe.
    """
    def __init__(self, stop, idle_minutes=IDLE_MINUTES):
        self.stop, self.idle_minutes, self.done = stop, idle_minutes, threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        # Training starts only past the idle threshold, and idle time only grows without input. So being under it
        # means there was input since, however brief and whenever it happened between two looks.
        while not self.done.wait(2):
            if idle_seconds() < self.idle_minutes * 60 or on_battery():
                self.stop()
                return

    def __enter__(self):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # do not sleep during the run
        self.thread.start()

    def __exit__(self, *_):
        self.done.set()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


# --- One check -------------------------------------------------------------------------------------------------

class Status:
    """The readable record of what happened; also remembers what was already learned."""

    def __init__(self, folder):
        self.path = folder / 'status.json'
        try:
            value = json.loads(self.path.read_text(encoding='utf-8'))
            self.value = value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            self.value = {}

    def set(self, state, message, **more):
        self.value.update(state=state, message=message, checked=time.strftime('%Y-%m-%d %H:%M:%S'), **more)
        train_speech.write_json(self.path, self.value)


def train_arguments(args, inbox_manifest):
    return ['--manifest', str(args.manifest), str(inbox_manifest), '--base', str(args.base), '--root', str(args.root),
            '--if-changed', *TRAINING]


def run_training(arguments, log):
    """train_speech in this process, with its output in `log` (a scheduled run has no console)."""
    with log.open('w', encoding='utf-8', buffering=1) as out:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                return train_speech.main(arguments)
            except SystemExit as stop:
                return stop.code if isinstance(stop.code, int) else 1
            except BaseException:
                traceback.print_exc(file=out)
                return 1


def labels(clips):
    return {clip['id']: hashlib.sha256(clip['reference'].encode()).hexdigest() for clip in clips}


def check(args, now=None):
    """One scheduled check; returns the exit code. The caller holds the lock."""
    now = time.time() if now is None else now
    folder = args.root / 'idle'
    status = Status(folder)
    value = status.value
    if value.get('state') == 'training':                  # that run neither finished nor gave way: it was killed
        retry_later(value, value.get('pendingKey'), now)
        status.set('failed', 'the previous run ended unexpectedly')
    def give_way():
        status.set(INTERRUPTED, 'stopped because the computer is in use; tried again when it is idle')
        os._exit(GAVE_WAY)

    if args.verify_model and any((args.inbox / '.pending').glob('*.txt')):
        reason = None if args.now else blocked(args.root, args.idle_minutes)
        if reason:
            status.set('waiting', reason)
            return 0
        status.set('verifying', 'checking recorded transcripts with the second local speech model')
        with contextlib.nullcontext() if args.now else Watchdog(give_way, args.idle_minutes):
            accepted, skipped = verify_pending(args.inbox, args.verify_model)
        value['lastVerification'] = {'matched': accepted, 'disagreed': skipped}
    mark = signature(listing(args.inbox), args.manifest)
    if value.get('settled') == mark:
        status.set('waiting for recordings', 'nothing new in the inbox')
        return 0
    durations = dict(value.get('durations') or {})
    clips, ignored, settling = scan(args.inbox, args.manifest, durations, now)
    known = labels(clips)
    learned = value.get('learned') or {}
    pending = sorted(clip_id for clip_id, label in known.items() if learned.get(clip_id) != label)
    inbox = {'pairs': len(clips), 'newOrChanged': len(pending), 'ignored': ignored}
    status.value['durations'] = {clip['audio_sha256']: durations[clip['audio_sha256']] for clip in clips}
    if not pending:
        status.set('waiting for recordings', 'nothing new in the inbox', inbox=inbox, settled=None if settling else mark)
        return 0
    if settling:
        status.set('waiting', 'recordings are still being added', inbox=inbox)
        return 0
    key = hashlib.sha256(json.dumps([[i, known[i]] for i in pending]).encode()).hexdigest()
    if not args.now and value.get('pendingKey') == key and now < value.get('retryAfter', 0):
        status.set('waiting', 'the last run failed; it is tried again later', inbox=inbox)
        return 0
    if value.get('pendingKey') != key:
        value.update(pendingKey=key, failures=0, retryAfter=0)
    reason = None if args.now else blocked(args.root, args.idle_minutes)
    if reason:
        status.set('waiting', reason, inbox=inbox)
        return 0
    manifest = args.inbox / '.nuvra-manifest.json'
    before = train_speech.read_state(args.root).get('lastRun')
    train_speech.write_json(manifest, {'clips': clips})
    status.set('training', f'learning from {len(clips)} recordings ({len(pending)} new or changed)', inbox=inbox,
               started=time.strftime('%Y-%m-%d %H:%M:%S'))

    with contextlib.nullcontext() if args.now else Watchdog(give_way, args.idle_minutes):
        code = run_training(train_arguments(args, manifest), folder / 'last-training.log')
    if code == 4:
        status.set('waiting', 'another training run is active')
        return 0
    if code not in (0, 3):
        retry_later(value, key, now)
        status.set('failed', f'training failed (code {code}); see idle/last-training.log', lastExitCode=code)
        return 1
    last = train_speech.read_state(args.root).get('lastRun')
    ran = bool(last) and last != before
    model = exported_model(args.root, last) if ran and code == 0 else None
    if model:
        message = 'a new model is ready; choose it under Models > Browse'
    elif ran and code == 0:
        message = 'training finished, but its model or report cannot be found; see idle/last-training.log'
    elif ran:
        message = 'the new recordings did not improve the model on dev; nothing was exported'
    else:
        message = 'the last run already covered these recordings; there is no new model'
    status.set('finished', message, learned=known, settled=mark, failures=0, retryAfter=0, lastExitCode=code,
               lastReport=last, finished=time.strftime('%Y-%m-%d %H:%M:%S'), **({'latestModel': str(model)} if model else {}))
    return 0


def exported_model(root, report):
    """The model folder a run's report says it accepted, when both really exist."""
    try:
        value = json.loads((root / report).read_text(encoding='utf-8'))
        folder = root / 'models' / value['model']
        return folder if value['accepted'] is True and (folder / 'model.bin').is_file() else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def retry_later(value, key, now):
    value['failures'] = int(value.get('failures', 0)) + 1
    value['pendingKey'] = key
    value['retryAfter'] = now + min(24, 2 ** (value['failures'] - 1)) * 3600


def run(args):
    args.root.mkdir(parents=True, exist_ok=True)
    args.inbox.mkdir(parents=True, exist_ok=True)
    try:
        with train_speech.exclusive(args.root / 'idle'):
            try:
                return check(args)
            except train_speech.InputError as error:
                Status(args.root / 'idle').set('failed', str(error))
                return 2
            except Exception as error:                    # a scheduled check has no console to show this on
                Status(args.root / 'idle').set('failed', f'{type(error).__name__}: {error}')
                raise
    except train_speech.Busy:
        return 0                                          # the run that holds the lock does the work


# --- Registration ----------------------------------------------------------------------------------------------

def pythonw():
    """The windowless Python of this environment, so a check never opens a console."""
    path = Path(sys.executable).with_name('pythonw.exe')
    if not path.is_file():
        raise train_speech.InputError(f'{path} is missing; run this with the training environment')
    return path


def command_arguments(args, command):
    values = [str(Path(__file__).resolve()), command, '--root', str(args.root), '--inbox', str(args.inbox),
              '--manifest', str(args.manifest), '--base', str(args.base), '--idle-minutes', str(args.idle_minutes)]
    if args.verify_model:
        values.extend(('--verify-model', str(args.verify_model)))
    return subprocess.list2cmdline(values)


def task_xml(args, user):
    """A task for the current account only: its own token, no elevation, one instance, mains power."""
    return f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Retrains Nuvra's local speech model from corrected recordings when the computer is idle.</Description></RegistrationInfo>
  <Triggers><TimeTrigger><Repetition><Interval>PT{EVERY_MINUTES}M</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition>
    <StartBoundary>2026-01-01T00:00:00</StartBoundary><Enabled>true</Enabled></TimeTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>{escape(user)}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>true</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>true</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT6H</ExecutionTimeLimit>
    <Priority>7</Priority>
    <Hidden>true</Hidden>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author"><Exec><Command>{escape(str(pythonw()))}</Command><Arguments>{escape(command_arguments(args, 'run'))}</Arguments><WorkingDirectory>{escape(str(train_speech.ROOT))}</WorkingDirectory></Exec></Actions>
</Task>'''


SCHEDULER = r'''
$ErrorActionPreference = 'Stop'
try {
    $service = New-Object -ComObject Schedule.Service
    $service.Connect()
    $folder = $service.GetFolder('\')
    if ($env:NUVRA_TASK_ACTION -eq 'delete') { $folder.DeleteTask($env:NUVRA_TASK_NAME, 0) }
    elseif ($env:NUVRA_TASK_ACTION -eq 'find') { $null = $folder.GetTask($env:NUVRA_TASK_NAME) }
    else { $null = $folder.RegisterTask($env:NUVRA_TASK_NAME, $env:NUVRA_TASK_XML, [int]$env:NUVRA_TASK_FLAGS, $null, $null, 3) }
    Write-Output '0'
} catch {
    $e = $_.Exception
    while ($e.InnerException) { $e = $e.InnerException }
    Write-Output $e.HResult
    [Console]::Error.WriteLine($e.Message)
}
'''


def scheduler(action, xml='', flags=6):
    """HRESULT of one Task Scheduler call (0: done). Codes, not localized messages, tell the outcomes apart."""
    done = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', SCHEDULER],
                          capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
                          env={**os.environ, 'NUVRA_TASK_ACTION': action, 'NUVRA_TASK_NAME': TASK, 'NUVRA_TASK_XML': xml,
                               'NUVRA_TASK_FLAGS': str(flags)})
    try:
        return int(done.stdout.strip().splitlines()[-1]), done.stderr.strip()
    except (ValueError, IndexError):
        return -1, done.stderr.strip()


def current_user():
    done = subprocess.run(['whoami', '/user', '/fo', 'csv', '/nh'], capture_output=True, text=True, timeout=30,
                          stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    return done.stdout.strip().rsplit(',', 1)[1].strip('"')      # the SID: names can be renamed or non-Latin


def startup_entry(value=None, delete=False):
    """The fallback's entry in this account's startup programs; returns the current value when not changing it."""
    import winreg
    if value is not None:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.SetValueEx(key, TASK, 0, winreg.REG_SZ, value)
        return value
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                            winreg.KEY_READ | (winreg.KEY_SET_VALUE if delete else 0)) as key:
            if delete:
                winreg.DeleteValue(key, TASK)
                return None
            return winreg.QueryValueEx(key, TASK)[0]
    except FileNotFoundError:
        return None


def register(args):
    if args.verify_model and not (args.verify_model / 'model.bin').is_file():
        raise train_speech.InputError('the second local speech model is missing model.bin')
    args.inbox.mkdir(parents=True, exist_ok=True)
    code, message = scheduler('create', task_xml(args, current_user()))
    if code == 0:
        startup_entry(delete=True)                        # never both
        print(f'Registered "{TASK}" in Task Scheduler for this account (every {EVERY_MINUTES} minutes).')
    elif code == ACCESS_DENIED:
        startup_entry(f'"{pythonw()}" ' + command_arguments(args, 'watch'))
        subprocess.Popen(f'"{pythonw()}" ' + command_arguments(args, 'watch'), cwd=train_speech.ROOT, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))
        print('Task Scheduler denied access, so it was added to this account\'s startup programs instead.')
    else:
        raise train_speech.InputError(f'Task Scheduler refused the task ({code}): {message}')
    config = paths.DATA / 'speech-learning.json'
    if args.capture_recordings or args.verify_model:
        train_speech.write_json(config, {'inbox': str(args.inbox), 'automaticLabels': bool(args.verify_model),
                                         'verifyModel': str(args.verify_model) if args.verify_model else None})
    else:
        config.unlink(missing_ok=True)
    print(f'Recordings and transcripts inbox: {args.inbox}')
    return 0


def unregister(args):
    startup_entry(delete=True)                            # a running watcher notices and ends
    code, message = scheduler('find')
    if code == 0:
        code, message = scheduler('delete')
    if code not in (0, NOT_FOUND, ACCESS_DENIED):
        raise train_speech.InputError(f'could not remove the task ({code}): {message}')
    if code == ACCESS_DENIED and scheduler('find')[0] == 0:
        raise train_speech.InputError('Task Scheduler denied removing the task')
    (paths.DATA / 'speech-learning.json').unlink(missing_ok=True)
    print('Automatic retraining is off. Recordings, models and reports were kept.')
    return 0


def watch(args):
    """The startup fallback: check regularly for as long as the startup entry exists.

    Each check runs in its own windowless process, because giving way to the user ends that process.
    """
    expected = f'"{pythonw()}" ' + command_arguments(args, 'watch')
    registration = hashlib.sha256(expected.encode()).hexdigest()
    try:
        with train_speech.exclusive(args.root / 'idle' / 'watch' / registration):
            while startup_entry() == expected:
                # run() holds the common training lock even while two registrations are handing over.
                subprocess.run(f'"{pythonw()}" ' + command_arguments(args, 'run'), cwd=train_speech.ROOT,
                               stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
                for _ in range(EVERY_MINUTES * 6):
                    time.sleep(10)
                    if startup_entry() != expected:
                        break
    except train_speech.Busy:
        pass
    return 0


def show(args):
    value = Status(args.root / 'idle').value
    registered = scheduler('find')[0] == 0
    print(f"Registered: {'Task Scheduler' if registered else 'startup programs' if startup_entry() else 'no'}")
    print(f"Inbox:      {args.inbox}")
    if args.verify_model:
        print(f"Verifier:   {args.verify_model}")
    for label, key in (('State', 'state'), ('Detail', 'message'), ('Checked', 'checked'), ('Finished', 'finished'),
                       ('Model', 'latestModel'), ('Report', 'lastReport')):
        if value.get(key) and (key != 'latestModel' or Path(value[key]).is_dir()):
            print(f'{label + ":":11} {value[key]}')
    if value.get('retryAfter', 0) > time.time():
        print('Retry:      ' + time.strftime('%Y-%m-%d %H:%M', time.localtime(value['retryAfter'])))
    inbox = value.get('inbox') or {}
    if inbox:
        print(f"Pairs:      {inbox['pairs']} usable, {inbox['newOrChanged']} new or changed at the last scan")
        for item in inbox['ignored']:
            print(f"  ignored {item['file']}: {item['why']}")
    print(f"Details:    {args.root / 'idle' / 'status.json'}")
    return 0


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('register', 'unregister', 'status', 'run', 'watch'))
    parser.add_argument('--root', type=Path, default=train_speech.LEARNING)
    parser.add_argument('--inbox', type=Path, help='folder of recordings and transcripts (default: <root>/inbox)')
    parser.add_argument('--manifest', type=Path, help='public corpus (default: <root>/corpus/manifest.json)')
    parser.add_argument('--base', type=Path, help='base checkpoint (default: <root>/base)')
    parser.add_argument('--capture-recordings', action='store_true', help='register this inbox for corrected in-app dictation')
    parser.add_argument('--verify-model', type=Path, help='also learn from unedited dictation agreed by this second local Whisper model')
    parser.add_argument('--idle-minutes', type=int, default=IDLE_MINUTES, help='how long nobody must have used the computer')
    parser.add_argument('--now', action='store_true', help='with run: do not wait for idle, power or a free GPU')
    args = parser.parse_args(argv)
    if not 1 <= args.idle_minutes <= 1440:
        parser.error('--idle-minutes must be between 1 and 1440')
    args.root = args.root.resolve()
    args.inbox = (args.inbox or args.root / 'inbox').resolve()
    args.manifest = (args.manifest or args.root / 'corpus' / 'manifest.json').resolve()
    args.base = (args.base or args.root / 'base').resolve()
    if not args.verify_model and args.command in ('run', 'status'):
        try:
            config = json.loads((paths.DATA / 'speech-learning.json').read_text(encoding='utf-8'))
            if config['automaticLabels'] is True and Path(config['inbox']).resolve() == args.inbox:
                args.verify_model = Path(config['verifyModel'])
        except (OSError, ValueError, KeyError, TypeError):
            pass  # capture is optional for a manually populated inbox
    if args.verify_model:
        args.verify_model = args.verify_model.resolve()
    return args


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args = parse(argv)
    if sys.platform != 'win32':
        print('Error: idle retraining needs Windows', file=sys.stderr)
        return 2
    try:
        return {'register': register, 'unregister': unregister, 'status': show, 'run': run, 'watch': watch}[args.command](args)
    except (train_speech.InputError, OSError, subprocess.SubprocessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
