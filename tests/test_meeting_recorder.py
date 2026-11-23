"""The fake WASAPI host never enumerates or opens workstation devices."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from arabic_relay import meetings, paths


def read_pcm(path):
    with wave.open(str(path), 'rb') as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 16000)
        return np.frombuffer(clip.readframes(clip.getnframes()), dtype='<i2')


class MeetingTracks(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)

    def test_long_track_is_chunked_without_lost_samples_and_long_silence_uses_no_extra_files(self):
        track = meetings.Track('system', 'test output', 16000, 2)
        rows = []
        track.folder, track.on_chunk = self.folder, rows.append
        self.addCleanup(track.abandon)
        stereo = np.tile(np.array([0.25, 0.5], dtype=np.float32), 16000).tobytes()
        for second in range(70):
            track.feed(stereo, second + 1)
        track.feed(stereo, 3603)
        track.append(track.convert(None))
        track.finish()
        self.assertGreaterEqual(len(rows), 4)
        self.assertEqual(sum(row['durationMs'] for row in rows), 71000)
        self.assertEqual(rows[-1]['startMs'], 3602000)
        self.assertTrue(all(0 < row['durationMs'] <= 25000 for row in rows))
        previous_end = 0
        for row in rows:
            if row != rows[-1]:
                self.assertEqual(row['startMs'], previous_end)
            pcm = read_pcm(self.folder / row['file'])
            self.assertTrue(np.all(pcm == 12288), 'stereo average or sample continuity changed')
            previous_end = row['startMs'] + row['durationMs']
        self.assertEqual(list(self.folder.glob('*.part')), [])

    def test_interrupted_raw_part_is_recovered_once_without_losing_even_samples(self):
        track = meetings.Track('mic', 'test microphone', 16000, 1)
        track.folder, track.on_chunk = self.folder, Mock()
        expected = np.arange(16000, dtype='<i2')
        track.append(expected.tobytes())
        track.abandon()
        part, = self.folder.glob('*.part')
        with part.open('ab') as handle:
            handle.write(b'x')
        info = {'version': 1, 'id': self.folder.name, 'state': 'recording', 'seconds': 0,
                'startedAt': 1700000000, 'sources': {'mic': {'seconds': 0}}, 'chunks': []}
        (self.folder / 'session.json').write_text(json.dumps(info), encoding='utf-8')
        meetings.recover_session(self.folder)
        saved = (self.folder / 'session.json').read_bytes()
        final = json.loads(saved)
        self.assertEqual(final['state'], 'interrupted')
        self.assertTrue(final['partial'])
        self.assertEqual(len(final['chunks']), 1)
        np.testing.assert_array_equal(read_pcm(self.folder / final['chunks'][0]['file']), expected)
        self.assertFalse(part.exists())
        meetings.recover_session(self.folder)
        self.assertEqual((self.folder / 'session.json').read_bytes(), saved)


@unittest.skipUnless(os.name == 'nt', 'WASAPI sessions use a Windows interprocess lock')
class WindowsMeetingControl(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.inbox = self.folder / 'inbox'
        self.inbox.mkdir()
        self.profile = self.folder / 'profile'
        self.profile.mkdir()
        self.enterContext(patch.object(paths, 'DATA', self.profile))
        self.enterContext(patch.object(meetings, 'SESSION', None))
        self.enterContext(patch.object(meetings, 'CLOSED', False))
        self.enterContext(patch.object(meetings, 'installed', return_value=True))
        self.enterContext(patch.object(meetings.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**10)))
        self.host = Mock()
        self.host.get_default_wasapi_device.return_value = {
            'maxInputChannels': 1, 'name': 'simulated mic', 'defaultSampleRate': 16000, 'index': 1}
        self.host.get_default_wasapi_loopback.return_value = {
            'maxInputChannels': 2, 'name': 'simulated speaker loopback', 'defaultSampleRate': 48000, 'index': 2}
        self.streams = {}
        self.host.open.side_effect = self.open_stream
        api = SimpleNamespace(paFloat32=1, paInputOverflow=2, paContinue=0, paComplete=1,
                              PyAudio=Mock(return_value=self.host))
        self.audio = self.enterContext(patch.object(meetings, 'audio', return_value=api))
        self.addCleanup(self.finish)

    def finish(self):
        if meetings.SESSION is not None:
            meetings.stop()
            meetings.SESSION.thread.join(3)

    def open_stream(self, **options):
        stream = Mock()
        stream.is_active.return_value = True
        self.streams[options['input_device_index']] = (stream, options['stream_callback'])
        return stream

    def enable(self):
        (self.profile / 'speech-learning.json').write_text(
            json.dumps({'inbox': str(self.inbox), 'automaticLabels': False}), encoding='utf-8')

    def test_no_capture_or_archive_until_explicit_start_in_registered_profile(self):
        self.assertFalse(meetings.status()['available'])
        with self.assertRaises(ValueError):
            meetings.start(self.folder / 'model')
        self.enable()
        self.assertTrue(meetings.status()['available'])
        meetings.recover()
        meetings.stop()
        self.audio.assert_not_called()
        self.assertFalse((self.folder / 'meetings').exists())

    def test_manual_start_records_both_sources_and_stop_keeps_only_pre_stop_audio(self):
        self.enable()
        started = meetings.start(self.folder / 'model')
        self.assertEqual(started['state'], 'recording')
        self.assertTrue(meetings.active())
        with self.assertRaises(ValueError):
            meetings.start(self.folder / 'model')
        self.assertEqual(self.host.open.call_count, 2)
        for device, channels, rate in [(1, 1, 16000), (2, 2, 48000)]:
            frames = rate // 2
            audio = np.full((frames, channels), 0.25, dtype=np.float32).tobytes()
            stream, callback = self.streams[device]
            stream.start_stream.assert_called_once()
            self.assertEqual(callback(audio, frames, {}, 0)[1], 0)
        self.finish()
        self.assertFalse(meetings.active())
        self.assertEqual(meetings.stop()['state'], 'stopped')
        self.assertEqual(callback(audio, frames, {}, 0)[1], 1)
        session = self.folder / 'meetings' / started['id']
        manifest = json.loads((session / 'session.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['state'], 'finished')
        self.assertEqual({c['source'] for c in manifest['chunks']}, {'mic', 'system'})
        for row in manifest['chunks']:
            pcm = read_pcm(session / row['file'])
            self.assertEqual(len(pcm), 8000)
            self.assertGreater(int(pcm.max()), 8000)
        self.assertEqual(list(session.glob('*.part')), [])
        self.host.terminate.assert_called_once()

    def test_available_microphone_continues_when_loopback_device_cannot_open(self):
        self.enable()
        self.host.get_default_wasapi_loopback.side_effect = OSError('simulated missing device')
        with self.assertLogs(level='ERROR'):
            started = meetings.start(self.folder / 'model')
        self.assertEqual(started['state'], 'recording')
        self.assertTrue(started['partial'])
        self.assertTrue(started['sources']['system']['error'])
        self.streams[1][1](np.ones(8000, dtype=np.float32).tobytes(), 8000, {}, 0)
        self.finish()
        folder = self.folder / 'meetings' / started['id']
        metadata = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
        self.assertEqual([r['source'] for r in metadata['chunks']], ['mic'])
        self.assertTrue(metadata['partial'])
