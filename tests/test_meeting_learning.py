"""Exercise meeting archives using generated PCM, a temporary profile, and scripted recognitions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from arabic_relay import paths
from scripts import retrain_when_idle as idle
from tests.test_speech_latency import wav


class MeetingArchive(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inbox = self.root / 'inbox'
        self.inbox.mkdir()
        self.foreground = self.root / 'large-v3-turbo-egyptian'
        self.verifier = self.root / 'large-v3'
        for folder, weights in [(self.foreground, b'egyptian'), (self.verifier, b'independent full model')]:
            folder.mkdir()
            (folder / 'model.bin').write_bytes(weights)
        self.enterContext(patch.object(paths, 'DATA', self.root / 'profile'))
        paths.DATA.mkdir()
        self.load = self.enterContext(patch.object(idle, 'load_verifier', side_effect=lambda folder: Path(folder)))
        self.answers = {}
        self.decode = self.enterContext(patch.object(idle, 'verify_transcript', side_effect=self.recognize))
        self.session = self.root / 'meetings' / '20261123-143000-abc123'
        self.session.mkdir(parents=True)
        self.state('finished')

    def state(self, state):
        data = {'version': 1, 'id': self.session.name, 'state': state, 'modelPath': str(self.foreground),
                'startedAt': 1795444200, 'chunks': []}
        (self.session / 'session.json').write_text(json.dumps(data), encoding='utf-8')

    def clip(self, source, seq, offset, value):
        path = self.session / f'{source}-{seq:06d}.wav'
        path.write_bytes(wav(value.to_bytes(2, 'little') * 16000))
        manifest = self.session / 'session.json'
        data = json.loads(manifest.read_text(encoding='utf-8'))
        data['chunks'].append({'file': path.name, 'source': source, 'index': seq,
                               'startMs': offset, 'durationMs': 1000})
        manifest.write_text(json.dumps(data), encoding='utf-8')
        return path

    def recognize(self, model, audio):
        return self.answers[Path(audio).name][int(Path(model) == self.verifier)]

    def process(self):
        return idle.process_meetings(self.inbox, self.verifier, idle.meeting_work(self.inbox))

    def test_no_jobs_for_unfinished_audio(self):
        clip = self.clip('mic', 1, 0, 1)
        clip.rename(clip.with_suffix('.wav.partial'))
        self.assertFalse(list(idle.meeting_work(self.inbox)))

    def test_recording_session_is_not_transcribed_until_stopped(self):
        self.state('recording')
        self.clip('mic', 1, 0, 2000)
        self.assertFalse(idle.meeting_work(self.inbox))
        self.process()
        self.load.assert_not_called()
        self.assertEqual(list(self.inbox.iterdir()), [])

    def test_both_sources_are_transcribed_only_agreement_trains_and_a_retry_is_idempotent(self):
        mic = self.clip('mic', 1, 0, 5000)
        output = self.clip('system', 1, 800, 9000)
        self.answers = {mic.name: ('نبدأ مع JEV', 'نبدأ مع jev.'), output.name: ('ما تشغلش الملف', 'شغل الملف')}
        self.process()
        records = json.loads((self.session / 'transcription.json').read_text(encoding='utf-8'))['clips']
        self.assertIs(records[mic.name]['matched'], True)
        self.assertIs(records[output.name]['matched'], False)
        self.assertEqual(records[output.name]['text'], self.answers[output.name][0])
        transcript = (self.session / 'transcript.txt').read_text(encoding='utf-8')
        self.assertLess(transcript.index('نبدأ مع JEV'), transcript.index('ما تشغلش الملف'))
        pair, = self.inbox.glob('meeting-*.txt')
        self.assertEqual(pair.read_text(encoding='utf-8'), 'نبدأ مع JEV')
        self.assertEqual(pair.with_suffix('.wav').read_bytes(), mic.read_bytes())
        count = self.decode.call_count
        self.assertEqual(count, 4)
        self.process()
        self.assertEqual(self.decode.call_count, count)
        self.assertEqual(list(self.inbox.glob('meeting-*.txt')), [pair])

    def test_interrupted_processing_resumes_after_the_last_checkpoint_and_rebuilds_text_without_decoding(self):
        self.state('interrupted')
        first = self.clip('mic', 1, 0, 1000)
        second = self.clip('system', 1, 3100, 2000)
        self.answers = {first.name: ('الكلام الأول', 'الكلام الأول'), second.name: ('الكلام التاني', 'الكلام التاني')}

        def interrupted(model, audio):
            if Path(audio) == second:
                raise KeyboardInterrupt
            return self.recognize(model, audio)

        self.decode.side_effect = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.process()
        self.decode.reset_mock()
        self.decode.side_effect = self.recognize
        self.process()
        self.assertEqual([call.args[1] for call in self.decode.call_args_list], [second, second])
        transcript = self.session / 'transcript.txt'
        expected = transcript.read_text(encoding='utf-8')
        self.assertIn('[00:00:03]', expected)
        transcript.unlink()
        self.decode.reset_mock()
        self.process()
        self.decode.assert_not_called()
        self.assertEqual(transcript.read_text(encoding='utf-8'), expected)

    def test_corrupt_audio_does_not_block_next_recording(self):
        bad = self.clip('mic', 1, 0, 1000)
        bad.write_bytes(b'broken WAV')
        next_clip = self.clip('system', 1, 0, 4000)
        self.answers = {next_clip.name: ('كمل', 'كمل')}
        self.process()
        records = json.loads((self.session / 'transcription.json').read_text(encoding='utf-8'))['clips']
        self.assertFalse(records[bad.name]['matched'])
        self.assertTrue(records[next_clip.name]['matched'])
        self.assertEqual(self.decode.call_count, 2)

    def test_replaying_publication_after_a_lost_checkpoint_preserves_corrected_inbox_label(self):
        clip = self.clip('mic', 1, 0, 3000)
        self.answers = {clip.name: ('افتح الملف', 'افتح الملف')}
        self.process()
        label, = self.inbox.glob('meeting-*.txt')
        label.write_text('ما تفتحش الملف', encoding='utf-8')
        (self.session / 'transcription.json').unlink()
        self.process()
        self.assertEqual(label.read_text(encoding='utf-8'), 'ما تفتحش الملف')

    def test_invalid_checkpoint_entry_can_be_rebuilt_from_the_original_audio(self):
        clip = self.clip('system', 1, 2000, 1500)
        self.answers = {clip.name: ('هنا الصوت', 'هنا الصوت')}
        (self.session / 'transcription.json').write_text(json.dumps({'clips': {clip.name: None}}), encoding='utf-8')
        self.process()
        self.assertIn('هنا الصوت', (self.session / 'transcript.txt').read_text(encoding='utf-8'))

    def test_identical_model_can_generate_notes_but_cannot_approve_training_audio(self):
        clip = self.clip('mic', 1, 0, 4000)
        self.answers = {clip.name: ('كلام مسموع', 'كلام مسموع')}
        idle.process_meetings(self.inbox, self.foreground, idle.meeting_work(self.inbox))
        self.assertIn('كلام مسموع', (self.session / 'transcript.txt').read_text(encoding='utf-8'))
        self.assertEqual(self.decode.call_count, 1)
        self.assertEqual(list(self.inbox.iterdir()), [])
