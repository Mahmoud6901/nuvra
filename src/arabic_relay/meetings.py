"""Records a meeting to disk for later local learning: the microphone and what the computer plays (Windows WASAPI).

Nothing records until start() is called, and what arrives after stop() is thrown away. Each source is its own track in
<training inbox's parent>/meetings/<session>/, so nothing leaves the computer. A meeting of any length survives a crash:
- `mic-000001.wav`, `system-000001.wav`: finished pieces (16 kHz mono PCM, at most 25 seconds). The name appears only when
  the file is complete, and the file never changes afterwards;
- `.mic-000001-000000000.part`: the piece being recorded, raw, with its start in milliseconds. recover() finishes it;
- `session.json`: the state, the speech model in use and when each piece starts. Both tracks share one clock.
"""
import contextlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import sys
import threading
import time
import wave

from . import speech_learning

VERSION = 1
RATE = 16000
BYTES = 2                            # 16-bit mono
SAMPLES_PER_MS = RATE // 1000
MAX_SECONDS, CUT_AFTER_SECONDS = 25, 15   # a piece ends at the quietest spot between these
GAP_SECONDS, FILL_SECONDS = 0.5, 3.0      # Windows sends nothing while the computer plays nothing
BUFFERS_PER_SECOND, MAX_WAITING = 50, 6000   # a minute of both tracks, if the disk is slow
MIC_SILENT_SECONDS = 10              # a microphone always sends something; this long without is a lost device
CHECK_DISK_SECONDS = 10
START_SECONDS, STOP_SECONDS = 10, 10
SOURCES = ('mic', 'system')
PART = re.compile(r'\.(mic|system)-(\d{6})-(\d{9,})\.part')
LOCK_NAME = '.recording.lock'

UNAVAILABLE = 'تسجيل الاجتماعات مش متاح على الجهاز ده.'
NOT_SET_UP = 'فعّل التعلّم المحلي الأول علشان تسجّل اجتماع.'
NO_SOURCE = 'اختار الميكروفون أو صوت الجهاز.'
BUSY = 'فيه تسجيل اجتماع شغّال بالفعل.'
NOT_READY = 'ما قدرتش أبدأ تسجيل الاجتماع.'
CLOSING = 'البرنامج بيقفل.'
LOW_DISK = 'مساحة القرص مش كفاية، فالتسجيل وقف. اللي اتسجّل محفوظ.'
NOT_SAVED = 'ما قدرتش أحفظ التسجيل، فوقف. اللي اتحفظ موجود.'
STOPPED_EARLY = 'البرنامج قفل قبل ما التسجيل يخلص.'
NO_DEVICE = {'mic': 'ما قدرتش أفتح الميكروفون.', 'system': 'ما قدرتش أفتح صوت الجهاز.'}
LOST_DEVICE = {'mic': 'الميكروفون اتفصل أثناء التسجيل.', 'system': 'صوت الجهاز اتفصل أثناء التسجيل.'}

CONTROL = threading.Lock()           # one start at a time
SESSION = None                       # the current or last recording
CLOSED = False                       # the app is closing


def audio():
    """The WASAPI library, loaded only to record."""
    import pyaudiowpatch
    return pyaudiowpatch


def root():
    current = speech_learning.settings()
    return None if current is None else current['inbox'].parent / 'meetings'


# --- One recording at a time, across processes -----------------------------------------------------------------

def acquire(folder):
    """An open file that holds the recording lock, or None while another recording has it. Windows frees it on a crash."""
    import msvcrt
    handle = open(folder / LOCK_NAME, 'a+b')
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return handle
    except OSError:
        handle.close()
        return None


def active():
    """True while any process on this computer records: someone listening to a meeting does not look busy to Windows."""
    folder = root()
    if os.name != 'nt' or folder is None or not (folder / LOCK_NAME).is_file():
        return False
    try:
        handle = acquire(folder)
    except OSError:
        return True
    if handle is None:
        return True
    handle.close()
    return False


# --- Files -----------------------------------------------------------------------------------------------------

def chunk_name(source, index):
    return f'{source}-{index:06d}.wav'


def wav(data):
    out = io.BytesIO()
    with wave.open(out, 'wb') as clip:
        clip.setnchannels(1)
        clip.setsampwidth(BYTES)
        clip.setframerate(RATE)
        clip.writeframes(data)
    return out.getvalue()


def row(source, index, start, size):
    return {'file': chunk_name(source, index), 'source': source, 'index': index, 'startMs': start // SAMPLES_PER_MS,
            'durationMs': size // BYTES // SAMPLES_PER_MS}


def write_session(folder, value):
    speech_learning.write(folder / 'session.json', json.dumps(value, ensure_ascii=False, indent=1).encode())


def quietest(data):
    """Where to end a full piece, in bytes: a word cut in two would be recognized wrongly on both sides."""
    import numpy as np
    step = RATE // 50
    tail = np.frombuffer(data, np.int16)[CUT_AFTER_SECONDS * RATE:].astype(np.float32).reshape(-1, step)
    loudness = np.convolve((tail ** 2).mean(axis=1), np.ones(10), 'valid')      # 200 ms
    last = len(loudness) - 1 - int(loudness[::-1].argmin())                     # the latest of equally quiet spots
    return (CUT_AFTER_SECONDS * RATE + (last + 5) * step) * BYTES


class Track:
    """One source: turns device buffers into finished pieces. Used by the recording thread only."""

    def __init__(self, source, device, rate, channels):
        self.source, self.device, self.rate, self.channels = source, device, rate, channels
        self.folder = self.on_chunk = None
        self.index, self.start = 1, 0             # the piece being recorded: its number and first sample
        self.buffer, self.part = bytearray(), None
        self.samples, self.chunks = 0, 0          # heard, and pieces written
        self.resampler, self.heard, self.dropouts = None, False, 0
        self.recording, self.error, self.stream, self.last = True, None, None, None

    def status(self):
        return {'device': self.device, 'seconds': round(self.samples / RATE, 1), 'chunks': self.chunks,
                'recording': self.recording, 'dropouts': self.dropouts, 'error': self.error}

    def feed(self, raw, at):
        """A buffer of 32-bit float frames that ended `at` seconds into the session."""
        import numpy as np
        frames = np.frombuffer(raw, np.float32).reshape(-1, self.channels)
        if self.source == 'system' and self.channels > 2:
            mono = frames[:, :2].mean(axis=1) + 0.7 * frames[:, 2]    # surround: speech can be in the center
        else:
            mono = frames.mean(axis=1)
        begin = max(0, int((at - len(mono) / self.rate) * RATE))
        late = begin - self.start - len(self.buffer) // BYTES
        if not self.heard or late > GAP_SECONDS * RATE:
            if self.heard and self.source == 'mic':
                self.dropouts += 1           # a microphone never pauses by itself
            self.skip(late)
        self.heard = True
        self.append(self.convert(mono))

    def convert(self, mono):
        import av
        import numpy as np
        if self.resampler is None:
            if mono is None:
                return b''
            self.resampler = av.AudioResampler(format='s16', layout='mono', rate=RATE)
        frame = None
        if mono is not None:
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(mono, np.float32).reshape(1, -1), format='flt', layout='mono')
            frame.sample_rate = self.rate
        data = b''.join(out.to_ndarray().tobytes() for out in self.resampler.resample(frame))
        if mono is None:
            self.resampler = None
        return data

    def skip(self, samples):
        """Nothing arrived for a while: a short pause is silence, a long one ends the piece."""
        self.append(self.convert(None))
        if self.buffer and samples <= FILL_SECONDS * RATE:
            self.append(bytes(samples * BYTES))
        else:
            self.finish()
            self.start += max(0, samples) // SAMPLES_PER_MS * SAMPLES_PER_MS

    def append(self, data):
        self.samples += len(data) // BYTES
        limit = MAX_SECONDS * RATE * BYTES
        while data:
            taken, data = data[:limit - len(self.buffer)], data[limit - len(self.buffer):]
            if self.part is None:
                self.part = self.open_part(self.index, self.start)
            self.part.write(taken)
            self.part.flush()
            self.buffer += taken
            if len(self.buffer) == limit:
                self.cut()

    def open_part(self, index, start):
        return open(self.folder / f'.{self.source}-{index:06d}-{start // SAMPLES_PER_MS:09d}.part', 'wb')

    def cut(self):
        at = quietest(self.buffer)
        head, tail, start = bytes(self.buffer[:at]), self.buffer[at:], self.start + at // BYTES
        following = self.open_part(self.index + 1, start)      # the rest is safe before the finished piece appears
        try:
            following.write(tail)
            following.flush()
            self.publish(head)
        except BaseException:
            discard(following)
            raise
        done, self.part, self.buffer = self.part, following, tail
        self.index, self.start = self.index + 1, start
        discard(done)

    def finish(self):
        """Ends the piece being recorded. On failure its .part stays for recover()."""
        if self.part is None:
            return
        self.publish(bytes(self.buffer))
        done, self.part = self.part, None
        self.start += len(self.buffer) // BYTES // SAMPLES_PER_MS * SAMPLES_PER_MS
        self.buffer = bytearray()
        self.index += 1
        discard(done)

    def publish(self, data):
        if data.count(0) == len(data):       # nothing was played, or the microphone is muted
            return
        speech_learning.write(self.folder / chunk_name(self.source, self.index), wav(data))
        self.chunks += 1
        self.on_chunk(row(self.source, self.index, self.start, len(data)))

    def abandon(self):
        if self.part is not None:
            with contextlib.suppress(OSError):
                self.part.close()


def discard(part):
    try:
        part.close()
        os.unlink(part.name)
    except OSError:                          # a scanner holds it; recover() clears it
        logging.info('Meetings: a part file is in use')


# --- One recording ---------------------------------------------------------------------------------------------

class Session:
    def __init__(self, parent, model, sources):
        self.parent, self.model, self.sources = parent, str(model), sources
        self.id = time.strftime('%Y%m%d-%H%M%S-') + secrets.token_hex(3)
        self.folder = parent / self.id
        self.state, self.error, self.partial = 'starting', None, False
        self.started = self.finished = None
        self.began = False                   # at least one device really started
        self.zero = 0.0                      # monotonic time of the shared clock's start
        self.tracks, self.rows, self.missing = [], [], {}
        self.buffers = queue.Queue(MAX_WAITING)
        self.halt, self.ready, self.done = threading.Event(), threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self.run, name='meeting-recorder', daemon=True)

    # What the page and the disk see

    def parts(self):
        return {**self.missing, **{track.source: track.status() for track in self.tracks}}

    def seconds(self):
        return 0.0 if self.started is None else round((self.finished or time.time()) - self.started, 1)

    def status(self):
        state = 'stopping' if self.halt.is_set() and self.state in ('starting', 'recording') else self.state
        partial = self.partial or any(track.dropouts for track in self.tracks)
        return {'state': state, 'id': self.id, 'startedAt': self.started, 'seconds': self.seconds(), 'partial': partial,
                'error': self.error, 'sources': self.parts()}

    def save(self):
        state = {'recording': 'recording', 'error': 'error'}.get(self.state, 'finished')
        write_session(self.folder, {
            'version': VERSION, 'id': self.id, 'state': state, 'modelPath': self.model, 'pid': os.getpid(),
            'startedAt': self.started, 'finishedAt': self.finished, 'seconds': self.seconds(), 'sampleRate': RATE,
            'channels': 1, 'partial': self.status()['partial'], 'error': self.error, 'sources': self.parts(),
            'chunks': self.rows})

    def note(self, error):
        self.partial, self.error = True, error

    # The recording thread owns the devices: Windows audio must be opened and closed by the same thread

    def run(self):
        lock = host = None
        try:
            lock = self.lock()
            recover_sessions(self.parent)
            api = audio()
            host = api.PyAudio()
            self.open(api, host)
            self.begin()
            self.record()
        except ValueError as error:
            self.fail(str(error))
        except OSError:
            logging.exception('A meeting recording failed')
            self.fail(NOT_READY if self.started is None else NOT_SAVED)
        except Exception:
            logging.exception('A meeting recording failed')
            self.fail(NOT_READY)
        finally:
            # The pieces first: closing a broken device can hang.
            self.halt.set()
            try:
                self.close()
            except Exception:
                logging.exception('A meeting recording could not be finished')
            if lock is not None:
                lock.close()
            self.ready.set()
            self.done.set()
            for track in self.tracks:
                if track.stream is not None:
                    with contextlib.suppress(Exception):
                        track.stream.close()
            if host is not None:
                with contextlib.suppress(Exception):
                    host.terminate()

    def lock(self):
        self.parent.mkdir(exist_ok=True)
        if speech_learning.is_link(self.parent):
            raise ValueError(NOT_READY)
        for _ in range(10):                  # active() in another process holds it for a moment
            if (lock := acquire(self.parent)) is not None:
                return lock
            time.sleep(0.1)
        raise ValueError(BUSY)

    def open(self, api, host):
        for source in self.sources:
            try:
                info = host.get_default_wasapi_device(d_in=True) if source == 'mic' else host.get_default_wasapi_loopback()
                rate, channels = int(info['defaultSampleRate']), int(info['maxInputChannels'])
                track = Track(source, info['name'], rate, channels)
                track.stream = host.open(format=api.paFloat32, channels=channels, rate=rate, input=True, start=False,
                                         input_device_index=info['index'], frames_per_buffer=rate // BUFFERS_PER_SECOND,
                                         stream_callback=self.listener(track, api))
                self.tracks.append(track)
            except (OSError, LookupError, ValueError):
                logging.exception('Meetings: %s could not be opened', source)
                self.missing[source] = {'device': None, 'seconds': 0.0, 'chunks': 0, 'recording': False, 'dropouts': 0,
                                        'error': NO_DEVICE[source]}
                self.note(NO_DEVICE[source])
        if not self.tracks:
            raise ValueError(self.error)

    def listener(self, track, api):
        def heard(data, frames, timing, flags):      # on the device's own thread; must never raise
            if self.halt.is_set() or not track.recording:
                return None, api.paComplete
            try:
                self.buffers.put_nowait((track, data, time.monotonic() - self.zero))
            except queue.Full:
                track.dropouts += 1
            if flags & api.paInputOverflow:
                track.dropouts += 1
            return None, api.paContinue
        return heard

    def begin(self):
        if self.halt.is_set():
            raise ValueError(NOT_READY)
        self.folder.mkdir()
        self.started, self.zero, self.state = time.time(), time.monotonic(), 'recording'
        for track in self.tracks:
            track.folder, track.on_chunk, track.last = self.folder, self.add, self.zero
        self.save()
        for track in self.tracks:
            try:
                track.stream.start_stream()
            except OSError:
                logging.exception('Meetings: %s did not start', track.source)
                self.lose(track, NO_DEVICE[track.source])
        if not any(track.recording for track in self.tracks):
            raise ValueError(self.error)
        self.began = True
        self.ready.set()

    def add(self, chunk):
        self.rows.append(chunk)
        try:
            self.save()
        except OSError:                      # the piece is safe, and the next save has its row
            logging.exception('Meetings: session.json is in use')

    def record(self):
        checked = time.monotonic()
        while not self.halt.is_set():
            try:
                self.take(self.buffers.get(timeout=0.25))
            except queue.Empty:
                pass
            now = time.monotonic()
            for track in self.tracks:
                if track.recording and not self.alive(track, now):
                    self.drain()
                    self.lose(track, LOST_DEVICE[track.source])
            if not any(track.recording for track in self.tracks):
                self.fail(self.error)
                return
            if now - checked >= CHECK_DISK_SECONDS:
                checked = now
                if shutil.disk_usage(self.folder).free < speech_learning.MIN_FREE_BYTES:
                    self.fail(LOW_DISK)
                    return

    def alive(self, track, now):
        try:
            if not track.stream.is_active():
                return False
        except OSError:
            return False
        # Buffers still waiting to be written are not silence.
        return track.source != 'mic' or not self.buffers.empty() or now - track.last < MIC_SILENT_SECONDS

    def take(self, item):
        track, data, at = item
        track.last = self.zero + at
        track.feed(data, at)

    def drain(self):
        with contextlib.suppress(queue.Empty):
            while True:
                self.take(self.buffers.get_nowait())

    def lose(self, track, error):
        track.recording, track.error = False, error
        self.note(error)
        track.append(track.convert(None))
        track.finish()
        self.save()

    def fail(self, error):
        self.state, self.error = 'error', error
        self.partial = self.started is not None

    def close(self):
        """Keeps everything heard before the stop, and says how it ended."""
        if self.started is None:             # nothing was recorded, so there is nothing to keep
            self.state = 'error'
            return
        saved = True
        try:
            self.drain()
        except Exception:
            logging.exception('Meetings: the last buffers could not be written')
            saved = False
        for track in self.tracks:
            track.recording = False
            try:
                track.append(track.convert(None))
                track.finish()
            except Exception:
                logging.exception('Meetings: a piece is left for recovery')
                track.abandon()
                saved = False
        if not saved:
            self.fail(NOT_SAVED)
        self.finished = time.time()
        if self.state != 'error':
            self.state = 'stopped'
        self.save()


# --- After a crash ---------------------------------------------------------------------------------------------

def recover():
    """Finishes what an app that stopped unexpectedly left behind. Does nothing while a meeting is being recorded."""
    folder = root()
    if os.name != 'nt' or folder is None or not folder.is_dir() or speech_learning.is_link(folder):
        return
    try:
        lock = acquire(folder)
        if lock is None:
            return
        with lock:
            recover_sessions(folder)
    except OSError:
        logging.exception('Meetings could not be recovered')


def recover_sessions(parent):
    for folder in sorted(parent.iterdir()):
        if folder.is_dir() and not speech_learning.is_link(folder):
            try:
                recover_session(folder)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                logging.exception('A meeting could not be recovered')


def recover_session(folder):
    if not (folder / 'session.json').is_file():
        return
    value = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
    parts = sorted((m[1], int(m[2]), int(m[3]), path) for path in folder.iterdir()
                   if (m := PART.fullmatch(path.name)) and path.is_file() and not path.is_symlink())
    if not parts and value['state'] != 'recording':
        return
    rows = {item['file']: item for item in value['chunks']}
    starts = [(source, index, start) for source, index, start, _ in parts]
    starts += [(item['source'], item['index'], item['startMs']) for item in rows.values()]
    done = []
    for source, index, start, path in parts:
        data, name = path.read_bytes(), chunk_name(source, index)
        if (folder / name).exists():         # stopped after the piece was finished and before its .part was removed
            with wave.open(str(folder / name), 'rb') as clip:
                data = bytes(clip.getnframes() * BYTES)
            rows.setdefault(name, row(source, index, start * SAMPLES_PER_MS, len(data)))
        else:
            # A stop in the middle of a cut leaves the rest in two files; it belongs to the later piece.
            later = [other - start for who, number, other in starts if who == source and number > index]
            size = min([len(data) // BYTES * BYTES, *(ms * SAMPLES_PER_MS * BYTES for ms in later)])
            data = data[:size]
            if data.count(0) != len(data):
                speech_learning.write(folder / name, wav(data))
                rows[name] = row(source, index, start * SAMPLES_PER_MS, len(data))
        done.append(path)
    chunks = sorted(rows.values(), key=lambda item: (item['startMs'], item['source']))
    ends = [item['startMs'] + item['durationMs'] for item in chunks]
    value.update(chunks=chunks, seconds=max([value['seconds'], *(end / 1000 for end in ends)]))
    for source, state in value['sources'].items():
        mine = [item for item in chunks if item['source'] == source]
        state.update(recording=False, chunks=len(mine),
                     seconds=round(max(state['seconds'], sum(item['durationMs'] for item in mine) / 1000), 1))
    if value['state'] == 'recording':
        value.update(state='interrupted', partial=True, error=STOPPED_EARLY, finishedAt=value['startedAt'] + value['seconds'])
    write_session(folder, value)
    for path in done:                        # last: an interrupted recovery starts over
        path.unlink(missing_ok=True)


# --- What the app calls ----------------------------------------------------------------------------------------

def installed():
    return sys.platform == 'win32' and importlib.util.find_spec('pyaudiowpatch') is not None


def status():
    session = SESSION
    current = ({'state': 'idle', 'id': None, 'startedAt': None, 'seconds': 0.0, 'partial': False, 'error': None, 'sources': {}}
               if session is None else session.status())
    return {'available': installed() and root() is not None, **current}


def start(model, microphone=True, speakers=True):
    """Starts recording the default microphone and/or what the default output plays. `model`: the speech model in use."""
    global SESSION
    if not installed():
        raise ValueError(UNAVAILABLE)
    sources = [source for source, wanted in zip(SOURCES, (microphone, speakers)) if wanted]
    if not sources:
        raise ValueError(NO_SOURCE)
    if not CONTROL.acquire(blocking=False):
        raise ValueError(BUSY)
    try:
        if SESSION is not None and SESSION.thread.is_alive():
            raise ValueError(BUSY)
        parent = root()
        if parent is None:
            raise ValueError(NOT_SET_UP)
        if shutil.disk_usage(parent.parent).free < speech_learning.MIN_FREE_BYTES:
            raise ValueError(LOW_DISK)
        session = Session(parent, Path(model), sources)
        SESSION = session                    # before CLOSED is read, so shutdown() stops it or it never starts
        if CLOSED:
            SESSION = None
            raise ValueError(CLOSING)
        session.thread.start()
        if not session.ready.wait(START_SECONDS):
            session.halt.set()               # a device that opens late finds this and records nothing
            raise ValueError(NOT_READY)
        if not session.began:
            session.thread.join(2)           # so the next try does not find this one still closing
            raise ValueError(session.error or NOT_READY)
        return status()
    finally:
        CONTROL.release()


def stop():
    """Ends the recording. Buffers that arrive from now on are dropped; what was heard before is finished and kept."""
    session = SESSION
    if session is not None:
        session.halt.set()
        if session.thread.is_alive():
            session.done.wait(STOP_SECONDS)
    return status()


def shutdown():
    """The app is closing: stops, and nothing starts again."""
    global CLOSED
    CLOSED = True
    return stop()
