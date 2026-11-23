from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from faster_whisper.audio import decode_audio

from scripts import meeting_audio
from tests.test_speech_latency import wav


class MeetingPlayback(TestCase):
    def test_playable_export_preserves_track_offsets_overlap_and_gaps(self):
        rate = 16000
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        chunks = []
        for source, start_ms, hz in [('mic', 250, 400), ('system', 750, 1000)]:
            signal = (8000 * np.sin(2 * np.pi * hz * np.arange(rate) / rate)).astype('<i2')
            name = source + '-000001.wav'
            (folder / name).write_bytes(wav(signal.tobytes()))
            chunks.append({'source': source, 'file': name, 'index': 1, 'startMs': start_ms, 'durationMs': 1000})
        meeting_audio.export_recording(folder, chunks, 2)
        audio = decode_audio(str(folder / 'meeting.m4a'), sampling_rate=rate)
        self.assertAlmostEqual(len(audio) / rate, 2, delta=0.08)

        def power_at(start, hz):
            window = audio[round(start * rate):round((start + 0.1) * rate)]
            return abs(np.dot(window, np.exp(-2j * np.pi * hz * np.arange(len(window)) / rate))) / len(window)

        for start, mic, system in [(0.05, False, False), (0.45, True, False), (0.95, True, True),
                                   (1.45, False, True), (1.85, False, False)]:
            with self.subTest(start=start):
                self.assertEqual(power_at(start, 400) > 0.012, mic)
                self.assertEqual(power_at(start, 1000) > 0.012, system)
        self.assertTrue(all((folder / row['file']).is_file() for row in chunks))
