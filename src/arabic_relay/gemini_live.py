"""Gemini 3.5 Transcribe Live: one recording streamed over Google's documented Live API WebSocket.

Used only while the user has chosen Gemini for dictation and saved their own API key. A recording is one push-to-talk
activity: activityStart, the microphone audio, activityEnd. While the user speaks the server sends interim previews, each
replacing the one before; they are shown and never used as text. The transcript is only what arrives as final
inputTranscription. The model need not say that the turn is complete, so a final after activityEnd followed by a short
quiet moment ends the recording.

Recordings stop at five minutes, half of the ten minutes a live session may last. If the connection drops anyway, a new
one gets this recording from its start, because finals carry no position that a second session could continue from.
Nothing is ever logged but a reason: no audio, text or key.
"""
import base64
import json
import logging
import re
import threading
import time

MODEL = 'gemini-3.5-transcribe-live'
URL = 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'
LANGUAGES = ['ar-EG', 'en-US']       # the model switches between them inside one sentence
RATE = 16000
MAX_SAMPLES = 300 * RATE
MAX_CHUNK = 2 * RATE                 # samples per upload from the page
MESSAGE = 1600 * 2                   # bytes per message to Gemini: 100 ms
OPEN_TIMEOUT = 10
SETUP_TIMEOUT = 10
RECONNECTS = 2
BACKLOG_PAUSE = .01                  # between messages while audio waits, so a replay is not one burst
IDLE_PAUSE = .05                     # how long one read waits for the server before new audio is looked for
QUIET = .7                           # seconds without another message after a final that ends the recording
UNCOVERED_QUIET = 4.0                # ... when the finals are much shorter than the preview: more text is on its way
SILENT_TIMEOUT = 8.0                 # after activityEnd, when nothing at all was heard
FINAL_TIMEOUT = 20.0                 # after activityEnd or the latest message, when a final is still owed
IDLE_EXPIRY = 45                     # seconds without audio before an abandoned recording is dropped

NO_CONNECTION = 'Gemini ما قدرش يكمّل. راجع النت وجرّب تاني.'
BAD_KEY = 'Gemini رفض المفتاح. غيّره من تبويب النماذج.'
NO_ACCESS = 'مفتاح Gemini ده مالوش صلاحية على نموذج التفريغ المباشر.'
LIMIT = 'Gemini وصل لحد الاستخدام في حسابك. جرّب بعد شوية.'
REFUSED = 'Gemini رفض التسجيل ده. جرّب تاني بعد شوية.'
NO_FINAL = 'Gemini ما رجّعش النص النهائي. جرّب تاني.'
CANCELLED = 'التسجيل اتلغى.'

# The WebSocket library writes request headers, so the key, at debug level. It gets a logger that never prints them.
SOCKET_LOG = logging.getLogger('arabic_relay.gemini_socket')
SOCKET_LOG.setLevel(logging.CRITICAL)


class Rejected(ValueError):
    """Gemini refused, or answered without a final transcript. Connecting again by itself cannot help."""


class Interrupted(Exception):
    """The connection failed or closed early. `code` is a fixed word for the log."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def refusal(code, reason):
    """The message for a definite refusal, or None when the connection just broke."""
    reason = (reason or '').casefold()
    if 'api key' in reason or 'api_key' in reason or code == 401:
        return BAD_KEY
    if 'quota' in reason or 'exhausted' in reason or 'billing' in reason or 'rate limit' in reason or code == 429:
        return LIMIT
    if code in (403, 404, 1008):
        return NO_ACCESS
    if code in (400, 1007):
        return REFUSED
    return None


def translate_failure(error):
    """Any connection error as Rejected or Interrupted. The server's own words are never shown or logged."""
    if isinstance(error, (Rejected, Interrupted)):
        return error
    closed = getattr(error, 'rcvd', None)
    code, reason = getattr(closed, 'code', None), getattr(closed, 'reason', '')
    response = getattr(error, 'response', None)
    if response is not None:
        code = getattr(response, 'status_code', None)
    message = refusal(code, reason)
    return Rejected(message) if message else Interrupted(f'closed {code}' if code else type(error).__name__)


def open_socket(key):
    """The seam tests replace: an object with send(text), recv(timeout) and close()."""
    from websockets.sync.client import ClientConnection, connect
    from websockets.uri import parse_uri

    endpoint = parse_uri(URL)

    def connection(sock, protocol, **options):
        # A redirected handshake must never send the API key or the recording to another endpoint.
        if protocol.uri != endpoint:
            raise Rejected(NO_CONNECTION)
        return ClientConnection(sock, protocol, **options)

    # Entered here and closed by session(): newer websockets expect a context manager.
    return connect(URL, additional_headers={'x-goog-api-key': key}, user_agent_header='Nuvra', compression=None,
                   open_timeout=OPEN_TIMEOUT, close_timeout=1, max_size=1 << 20, logger=SOCKET_LOG,
                   create_connection=connection).__enter__()


def setup_message(keywords=()):
    transcription = {'languageCodes': list(LANGUAGES)}
    if keywords:
        transcription['customVocabulary'] = list(keywords)
    return {'setup': {'model': 'models/' + MODEL, 'generationConfig': {'responseModalities': ['TEXT']},
                      'realtimeInputConfig': {'automaticActivityDetection': {'disabled': True}},
                      'inputAudioTranscription': transcription}}


def words(text):
    return len(re.findall(r'\w+', text))


class Transcript:
    """What one session has said: finals in order, and the preview that the next final will replace."""

    def __init__(self):
        self.finals = []
        self.interim = ''
        self.ended = None            # when activityEnd was sent
        self.owed = 0                # words in the preview when activityEnd was sent
        self.after_end = []          # finals since then
        self.last = None             # when the latest message arrived
        self.complete = False        # the server said the turn is over

    def text(self):
        return ' '.join(filter(None, (part.strip() for part in self.finals)))

    def preview(self):
        return ' '.join(filter(None, (self.text(), self.interim.strip())))

    def end(self, now):
        self.ended = self.last = now
        self.owed = words(self.interim)

    def take(self, content, now):
        """One serverContent. Returns whether anything changed."""
        self.last = now
        changed = False
        interim, final = content.get('interimInputTranscription'), content.get('inputTranscription')
        if isinstance(interim, dict):
            value = interim.get('text')
            self.interim = value if isinstance(value, str) else ''
            changed = True
        if isinstance(final, dict):
            value = final.get('text')
            value = value if isinstance(value, str) else ''      # an unset text is an empty final
            self.finals.append(value)
            self.interim = ''
            if self.ended is not None:
                self.after_end.append(value)
            changed = True
        if self.ended is not None and (content.get('turnComplete') or content.get('generationComplete')):
            self.complete = True
        return changed

    def result(self, now):
        """The transcript once it is final, None while waiting. Raises when only a preview ever came."""
        if self.ended is None:
            return None
        interim = self.interim.strip()
        pending = bool(interim) and interim not in self.text()     # a preview no final has replaced yet
        if self.after_end and not pending:
            covered = words(' '.join(self.after_end)) * 2 >= self.owed
            if self.complete and covered or now - self.last >= (QUIET if covered else UNCOVERED_QUIET):
                return self.text()
        if self.complete and not self.after_end and not pending and self.owed == 0 and now - self.last >= QUIET:
            return self.text()                        # nothing was said since the last final, or at all
        if not self.after_end and not pending and self.owed == 0 and now - self.last >= SILENT_TIMEOUT:
            return self.text()
        if now - self.last >= FINAL_TIMEOUT:
            raise Rejected(NO_FINAL)
        return None


class Job:
    """One recording sent to Gemini while it is spoken. Owned by its unguessable job ID, like a local streaming job."""
    online = True

    def __init__(self, job_id, key, keywords=(), inline=False, rules=None):
        self.id = job_id
        self.inline = inline
        self.rules = rules                 # approved vocabulary rules in force when the recording started
        self.keywords = list(keywords)
        self.key = key
        self.pcm = bytearray()
        self.next_seq = 0
        self.closed = False
        self.cancelled = threading.Event()
        self.provisional = ''
        self.text = None
        self.error = None
        self.connections = 0
        self.condition = threading.Condition()
        self.created = self.touched = time.monotonic()
        self.times = {'start': self.created}
        self.thread = threading.Thread(target=self.run, daemon=True, name='nuvra-gemini')
        self.thread.start()

    # --- The page's side ---------------------------------------------------------------------------------------

    def append(self, seq, payload):
        with self.condition:
            if self.error is not None:
                raise ValueError(self.error)
            if self.closed or self.cancelled.is_set():
                raise ValueError(CANCELLED)
            if type(seq) is not int or seq != self.next_seq or not payload or len(payload) % 2 or len(payload) > MAX_CHUNK * 2:
                raise ValueError('صوت التسجيل مش سليم.')
            if len(self.pcm) + len(payload) > MAX_SAMPLES * 2:
                raise ValueError('التسجيل لازم يكون أقل من خمس دقايق.')
            self.pcm += payload
            self.next_seq += 1
            self.touched = time.monotonic()
            self.condition.notify_all()
            return {'seq': seq, 'samples': len(self.pcm) // 2, 'provisional': self.provisional, 'partial': True}

    def finish(self, total_samples, last_seq, timeout=90, cancel_event=None):
        """Ends the recording and waits for Gemini's final text. Never returns a preview."""
        with self.condition:
            if self.cancelled.is_set():
                raise ValueError(CANCELLED)
            if self.error is None and (last_seq != self.next_seq - 1 or total_samples != len(self.pcm) // 2 or not self.pcm):
                self.error = 'صوت التسجيل مش سليم.'
            self.closed = True
            self.times['stop'] = time.monotonic()
            self.condition.notify_all()
            deadline = time.monotonic() + timeout
            while self.text is None and self.error is None:
                if self.cancelled.is_set() or (cancel_event is not None and cancel_event.is_set()):
                    self.cancelled.set()
                    raise ValueError('اتلغى التحويل.' if cancel_event is not None else CANCELLED)
                if time.monotonic() >= deadline:
                    self.error = 'التحويل خد وقت طويل. جرّب تاني.'
                    break
                self.condition.wait(.25)
            if self.error is not None:
                raise ValueError(self.error)
            return self.text

    def cancel(self):
        """The connection closes within one read pause, on its own thread; audio not sent yet is dropped here."""
        self.cancelled.set()
        with self.condition:
            self.pcm = bytearray()
            self.provisional = ''
            self.key = ''
            self.condition.notify_all()

    # --- Gemini's side -----------------------------------------------------------------------------------------

    def over(self):
        return self.cancelled.is_set() or self.error is not None

    def fail(self, message):
        with self.condition:
            if self.error is None:
                self.error = message
            self.condition.notify_all()

    def run(self):
        while not self.over():
            try:
                self.session()
                return
            except Exception as error:
                failure = translate_failure(error)
            if self.over():
                return
            if isinstance(failure, Rejected):
                logging.info('Gemini dictation ended without a transcript')
                self.fail(str(failure))
                return
            logging.info('Gemini dictation: connection lost (%s)', failure.code)
            if self.connections > RECONNECTS:
                self.fail(NO_CONNECTION)
                return
            self.cancelled.wait(.4 * self.connections)

    def receive(self, socket, timeout):
        """One server message as a dict, or None when none came in time."""
        try:
            raw = socket.recv(timeout=timeout)
        except TimeoutError:
            return None
        try:
            message = json.loads(raw)
        except ValueError:
            raise Interrupted('invalid message') from None
        if not isinstance(message, dict):
            raise Interrupted('invalid message')
        if message.get('error'):
            details = message['error'] if isinstance(message['error'], dict) else {}
            raise translate_error(details)
        return message

    def session(self):
        """One connection, from setup to the final text. Returns early when the recording is cancelled."""
        self.connections += 1
        with self.condition:
            key = self.key
        if not key:
            return                        # cancelled meanwhile
        socket = open_socket(key)
        try:
            socket.send(json.dumps(setup_message(self.keywords)))
            deadline = time.monotonic() + SETUP_TIMEOUT
            while True:
                if self.over():
                    return
                message = self.receive(socket, IDLE_PAUSE)
                if message is not None and 'setupComplete' in message:
                    break
                if time.monotonic() >= deadline:
                    raise Interrupted('setup timeout')
            socket.send(json.dumps({'realtimeInput': {'activityStart': {}}}))
            self.talk(socket)
        finally:
            try:
                socket.close()
            except Exception:
                pass                      # it is gone either way

    def talk(self, socket):
        from websockets.exceptions import ConnectionClosedOK
        transcript, sent = Transcript(), 0
        while not self.over():
            with self.condition:
                available, closed = len(self.pcm), self.closed
                size = min(MESSAGE, available - sent)
                payload = bytes(self.pcm[sent:sent + size]) if size == MESSAGE or (closed and size > 0) else b''
                abandoned = not closed and time.monotonic() - self.touched > IDLE_EXPIRY
            if abandoned:
                logging.info('Gemini dictation: abandoned recording discarded')
                self.fail(CANCELLED)
                return
            if payload:
                socket.send(json.dumps({'realtimeInput': {'audio': {
                    'data': base64.b64encode(payload).decode('ascii'), 'mimeType': f'audio/pcm;rate={RATE}'}}}))
                sent += len(payload)
            elif closed and transcript.ended is None and not self.over():
                socket.send(json.dumps({'realtimeInput': {'activityEnd': {}}}))
                transcript.end(time.monotonic())
            try:
                message = self.receive(socket, BACKLOG_PAUSE if available - sent >= MESSAGE else IDLE_PAUSE)
            except ConnectionClosedOK as error:
                # A clean close can immediately follow the last final, before the quiet timer expires.
                if not (error.rcvd and error.rcvd.code == 1000 and transcript.after_end and not transcript.interim.strip()):
                    raise
                transcript.complete = True
                message = None
            now = time.monotonic()
            content = message.get('serverContent') if message else None
            if isinstance(content, dict) and transcript.take(content, now):
                with self.condition:
                    if not self.over():
                        self.provisional = transcript.preview()
                        if self.provisional:
                            self.times.setdefault('firstPartial', now)
                        self.condition.notify_all()
            text = transcript.result(now)
            if text is not None:
                with self.condition:
                    if not self.over():
                        self.text = text
                    self.condition.notify_all()
                return


def translate_error(details):
    message = refusal(details.get('code'), str(details.get('message') or ''))
    return Rejected(message) if message else Interrupted('server error')
