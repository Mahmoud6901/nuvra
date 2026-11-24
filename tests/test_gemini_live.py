"""Exercise a dictation from PCM upload to final text without a microphone or a Gemini account."""
import base64
import json
import queue
import threading
import unittest
from unittest.mock import patch

from arabic_relay import gemini_live as live


class Socket:
    def __init__(self, final=None):
        self.messages = []
        self.incoming = queue.Queue()
        self.audio_received = threading.Event()
        self.activity_ended = threading.Event()
        self.closed = threading.Event()
        self.final = final

    def publish(self, **content):
        self.incoming.put(json.dumps({'serverContent': content}).encode())

    def send(self, raw):
        message = json.loads(raw)
        self.messages.append(message)
        if 'setup' in message:
            self.incoming.put('{"setupComplete": {}}')
        event = message.get('realtimeInput', {})
        if 'audio' in event:
            self.audio_received.set()
        if 'activityEnd' in event:
            self.activity_ended.set()
            if self.final is not None:
                self.publish(inputTranscription={'text': self.final})

    def recv(self, timeout):
        try:
            value = self.incoming.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed.set()

    def audio_frames(self):
        return [m['realtimeInput']['audio'] for m in self.messages if 'audio' in m.get('realtimeInput', {})]


class GeminiLiveTests(unittest.TestCase):
    pcm = bytes(range(256)) * 30
    final = 'إيه علاقة JEV بالموضوع؟'

    def job(self, *sockets, keywords=()):
        opens = patch.object(live, 'open_socket', side_effect=sockets)
        connect = opens.start()
        self.addCleanup(opens.stop)
        job = live.Job('dictation-test', 'synthetic-key', keywords=keywords)
        self.addCleanup(job.thread.join, 2)
        self.addCleanup(job.cancel)
        return job, connect

    def preview(self, job, text):
        with job.condition:
            self.assertTrue(job.condition.wait_for(lambda: job.provisional == text, timeout=2), job.provisional)

    def finish(self, job, cancel=None):
        return job.finish(len(self.pcm) // 2, 0, timeout=4, cancel_event=cancel)

    def test_preview_is_replaced_then_a_text_only_final_completes_the_recording(self):
        socket = Socket(self.final)
        job, connect = self.job(socket, keywords=['JEV'])
        job.append(0, self.pcm)
        self.assertTrue(socket.audio_received.wait(2))
        for interim in ['إيه الوقت', 'إيه علاقة JEV']:
            socket.publish(interimInputTranscription={'text': interim})
            self.preview(job, interim)
        self.assertEqual(self.finish(job), self.final)
        self.assertTrue(socket.closed.wait(2))
        connect.assert_called_once_with('synthetic-key')
        setup = socket.messages[0]['setup']
        self.assertEqual(setup['model'], 'models/gemini-3.5-transcribe-live')
        self.assertEqual(setup['inputAudioTranscription'], {'languageCodes': ['ar-EG', 'en-US'], 'customVocabulary': ['JEV']})
        self.assertEqual(setup['generationConfig']['responseModalities'], ['TEXT'])
        self.assertTrue(setup['realtimeInputConfig']['automaticActivityDetection']['disabled'])
        self.assertEqual(socket.messages[1], {'realtimeInput': {'activityStart': {}}})
        self.assertEqual(socket.messages[-1], {'realtimeInput': {'activityEnd': {}}})
        frames = socket.audio_frames()
        self.assertTrue(all(frame['mimeType'] == 'audio/pcm;rate=16000' for frame in frames))
        decoded = [base64.b64decode(frame['data'], validate=True) for frame in frames]
        self.assertTrue(all(0 < len(frame) <= 3200 for frame in decoded))
        self.assertEqual(b''.join(decoded), self.pcm)

    def test_preview_without_a_final_never_becomes_output(self):
        socket = Socket()
        job, _ = self.job(socket)
        job.append(0, self.pcm)
        self.assertTrue(socket.audio_received.wait(2))
        self.assertNotIn('customVocabulary', socket.messages[0]['setup']['inputAudioTranscription'])
        socket.publish(interimInputTranscription={'text': 'كلام لسه بيتغير'})
        self.preview(job, 'كلام لسه بيتغير')
        with patch.object(live, 'FINAL_TIMEOUT', .1), self.assertRaisesRegex(ValueError, live.NO_FINAL):
            self.finish(job)

    def test_cancel_while_waiting_for_final(self):
        socket = Socket()
        job, _ = self.job(socket)
        job.append(0, self.pcm)
        errors = []
        cancel = threading.Event()

        def wait_for_final():
            try:
                self.finish(job, cancel)
            except ValueError as error:
                errors.append(str(error))

        waiting = threading.Thread(target=wait_for_final)
        waiting.start()
        try:
            self.assertTrue(socket.activity_ended.wait(2))
        finally:
            cancel.set()
            waiting.join(2)
        self.assertFalse(waiting.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertTrue(socket.closed.wait(2))

    def test_interrupted_session_replays_audio_and_discards_its_transcript(self):
        first, second = Socket(), Socket(self.final)
        job, connect = self.job(first, second)
        job.append(0, self.pcm)
        self.assertTrue(first.audio_received.wait(2))
        first.publish(inputTranscription={'text': 'نص قديم'})
        self.preview(job, 'نص قديم')
        first.incoming.put(ConnectionError('offline'))
        self.assertTrue(second.audio_received.wait(3))
        self.assertEqual(self.finish(job), self.final)
        self.assertEqual(connect.call_count, 2)
        self.assertEqual(b''.join(base64.b64decode(frame['data']) for frame in second.audio_frames()), self.pcm)

    def test_key_or_quota_refusal_is_not_retried(self):
        for code in (401, 429):
            with self.subTest(code=code):
                socket = Socket()
                job, connect = self.job(socket)
                job.append(0, self.pcm)
                self.assertTrue(socket.audio_received.wait(2))
                socket.incoming.put(json.dumps({'error': {'code': code, 'message': 'synthetic-key'}}))
                with self.assertRaises(ValueError) as caught:
                    self.finish(job)
                self.assertNotIn('synthetic-key', str(caught.exception))
                connect.assert_called_once()

    def test_authentication_uses_header_only(self):
        with patch('websockets.sync.client.connect') as connect:
            live.open_socket('synthetic-key')
        (url,), options = connect.call_args
        self.assertEqual(url.split('/')[2], 'generativelanguage.googleapis.com')
        self.assertNotIn('synthetic-key', url)
        self.assertEqual(options['additional_headers']['x-goog-api-key'], 'synthetic-key')
        from types import SimpleNamespace
        from websockets.uri import parse_uri
        with self.assertRaises(live.Rejected):
            options['create_connection'](None, SimpleNamespace(uri=parse_uri('wss://example.invalid')))

    def test_final_survives_a_normal_websocket_close_before_the_quiet_timer(self):
        from websockets.sync.server import serve
        received = bytearray()

        def server_session(ws):
            ws.recv(timeout=3)  # setup
            ws.send('{"setupComplete":{}}')
            while True:
                event = json.loads(ws.recv(timeout=3))['realtimeInput']
                if 'activityEnd' in event:
                    ws.send(json.dumps({'serverContent': {'inputTranscription': {'text': self.final}}}))
                    return  # send a normal 1000 close immediately after the text
                if 'audio' in event:
                    received.extend(base64.b64decode(event['audio']['data']))
                    ws.send(json.dumps({'serverContent': {'interimInputTranscription': {'text': 'إيه علاقة'}}}))

        with serve(server_session, '127.0.0.1', 0) as server:
            serving = threading.Thread(target=server.serve_forever, daemon=True)
            serving.start()
            try:
                with patch.object(live, 'URL', f'ws://127.0.0.1:{server.socket.getsockname()[1]}'):
                    job = live.Job('local-transport', 'synthetic-key')
                    try:
                        job.append(0, self.pcm)
                        self.preview(job, 'إيه علاقة')
                        self.assertEqual(self.finish(job), self.final)
                        self.assertEqual(received, self.pcm)
                        self.assertEqual(job.connections, 1)
                    finally:
                        job.cancel()
                        job.thread.join(2)
            finally:
                server.shutdown()
                serving.join(2)
