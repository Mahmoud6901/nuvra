import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from arabic_relay import paths
from scripts import retrain_when_idle as idle


class LocalRecordingConsensus(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.inbox = Path(temp.name) / 'inbox'
        self.pending = self.inbox / '.pending'
        self.pending.mkdir(parents=True)
        self.model_folder = Path(temp.name) / 'independent-model'
        self.model = object()
        self.load = self.enterContext(patch.object(idle, 'load_verifier', return_value=self.model))
        self.decode = self.enterContext(patch.object(idle, 'verify_transcript'))
        self.duration = self.enterContext(patch.object(idle, 'audio_seconds', return_value=3.0))

    def stage(self, name, text, audio=b'test recording'):
        (self.pending / (name + '.webm')).write_bytes(audio)
        (self.pending / (name + '.txt')).write_text(text, encoding='utf-8')

    def verify(self):
        return idle.verify_pending(self.inbox, self.model_folder)

    def test_consensus_keeps_first_spelling_and_publishes_complete_audio_once(self):
        first = 'شَغَّل JEV، لو سمحت.'
        self.stage('note', first)
        self.decode.return_value = 'شغل jev لو سمحت'
        self.verify()
        self.assertEqual((self.inbox / 'note.txt').read_text(encoding='utf-8'), first)
        self.assertEqual((self.inbox / 'note.webm').read_bytes(), b'test recording')
        self.verify()
        self.load.assert_called_once_with(self.model_folder)
        self.decode.assert_called_once_with(self.model, self.pending / 'note.webm')

    def test_disagreement_on_negation_name_number_or_programming_symbol_never_trains(self):
        candidates = [
            ('ماتشغلش الملف', 'شغل الملف'),
            ('سلمها لـ JEV', 'سلمها لـ Jeff'),
            ('خلي القيمة 12.5', 'خلي القيمة 125'),
            ('استخدم C++', 'استخدم C'),
            ('اكتب .NET', 'اكتب NET'),
        ]
        for index, (proposal, _) in enumerate(candidates):
            self.stage(str(index), proposal)
        self.decode.side_effect = [second for _, second in candidates]
        self.verify()
        self.assertEqual(self.decode.call_count, len(candidates))
        self.assertEqual(idle.listing(self.inbox), {})

    def test_an_existing_human_label_wins_without_running_another_model(self):
        self.stage('note', 'شغل النص')
        (self.inbox / 'note.txt').write_text('شغل النسخة', encoding='utf-8')
        (self.inbox / 'note.webm').write_bytes(b'test recording')
        self.verify()
        self.assertEqual((self.inbox / 'note.txt').read_text(encoding='utf-8'), 'شغل النسخة')
        self.decode.assert_not_called()
        self.load.assert_not_called()

    def test_stale_or_missing_verifier_binding_cannot_approve_an_app_recording(self):
        for binding in (None, self.model_folder.with_name('a different model')):
            with self.subTest(binding=binding):
                audio = str(binding).encode()
                name = 'nuvra-' + hashlib.sha256(audio).hexdigest()
                self.stage(name, 'شوف الملف', audio)
                if binding:
                    (self.pending / (name + '.verifier')).write_text(str(binding), encoding='utf-8')
                self.verify()
                self.assertFalse((self.inbox / (name + '.txt')).exists())
        self.load.assert_not_called()
        self.decode.assert_not_called()

    def test_damaged_binding_does_not_block_other_queued_dictation(self):
        audio = b'damaged proposal'
        name = 'nuvra-' + hashlib.sha256(audio).hexdigest()
        self.stage(name, 'افتح الملف', audio)
        (self.pending / (name + '.verifier')).write_bytes(bytes([255, 254, 0, 1]))
        self.stage('z-ready', 'اقفل الباب')
        self.decode.return_value = 'اقفل الباب'
        self.verify()
        self.assertFalse((self.inbox / (name + '.txt')).is_file())
        self.assertTrue((self.inbox / 'z-ready.txt').is_file())
        self.decode.assert_called_once()

    def test_human_correction_arriving_mid_decode_wins_over_agreeing_models(self):
        self.stage('note', 'هما موافقين')

        def decode_and_correct(model, audio):
            (self.inbox / 'note.webm').write_bytes(audio.read_bytes())
            (self.inbox / 'note.txt').write_text('هما مش موافقين', encoding='utf-8')
            return 'هما موافقين'

        self.decode.side_effect = decode_and_correct
        self.verify()
        self.assertEqual((self.inbox / 'note.txt').read_text(encoding='utf-8'), 'هما مش موافقين')

    def test_unpaired_empty_and_unusable_audio_never_load_the_verifier(self):
        (self.pending / 'audio-only.wav').write_bytes(b'waiting')
        (self.pending / 'text-only.txt').write_text('باقي الصوت', encoding='utf-8')
        self.stage('empty-audio', 'بدون صوت', b'')
        self.stage('too-short', 'مش واضح')
        self.duration.return_value = 0.1
        self.verify()
        self.load.assert_not_called()
        self.assertEqual(idle.listing(self.inbox), {})


class CaptureRegistration(TestCase):
    def test_verifier_decodes_locally_without_receiving_the_proposed_words(self):
        folder, audio = Path('second-local-model'), Path('pending-example.wav')
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            iter([SimpleNamespace(text=' ما تفتحش'), SimpleNamespace(text=' الملف')]), None)
        with patch('faster_whisper.WhisperModel', return_value=transcriber) as constructor:
            model = idle.load_verifier(folder)
            text = idle.verify_transcript(model, audio)
        self.assertEqual(text, 'ما تفتحش الملف')
        self.assertEqual(constructor.call_args.args, (str(folder),))
        self.assertTrue(constructor.call_args.kwargs['local_files_only'])
        self.assertEqual(constructor.call_args.kwargs['device'], 'cpu')
        args, options = transcriber.transcribe.call_args
        self.assertEqual(args, (str(audio),))
        self.assertEqual((options.get('language'), options.get('task')), ('ar', 'transcribe'))
        self.assertFalse(options.get('initial_prompt') or options.get('hotwords'))

    def test_opt_in_modes_and_unregister_use_only_the_requested_local_inbox(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        profile = root / 'isolated profile'
        inbox = root / 'voice samples'
        verifier = root / 'independent verifier'
        verifier.mkdir()
        (verifier / 'model.bin').write_bytes(b'test')
        args = ['register', '--root', str(root), '--inbox', str(inbox)]
        self.enterContext(patch.object(paths, 'DATA', profile))
        self.enterContext(patch.object(idle, 'pythonw', return_value=root / 'pythonw.exe'))
        self.enterContext(patch.object(idle, 'current_user', return_value='test user'))
        scheduler = self.enterContext(patch.object(idle, 'scheduler', return_value=(0, '')))
        self.enterContext(patch.object(idle, 'startup_entry'))
        self.enterContext(patch('builtins.print'))
        config = profile / 'speech-learning.json'
        idle.register(idle.parse(args))
        self.assertFalse(config.exists())
        idle.register(idle.parse(args + ['--capture-recordings']))
        manual = json.loads(config.read_text(encoding='utf-8'))
        self.assertEqual(Path(manual['inbox']), inbox)
        self.assertIs(manual['automaticLabels'], False)
        idle.register(idle.parse(args + ['--capture-recordings', '--verify-model', str(verifier)]))
        automatic = json.loads(config.read_text(encoding='utf-8'))
        self.assertIs(automatic['automaticLabels'], True)
        self.assertEqual(Path(automatic['verifyModel']), verifier)
        self.assertIn('--verify-model', scheduler.call_args.args[1])
        (inbox / 'manual.txt').write_text('ابقى هنا', encoding='utf-8')
        idle.unregister(idle.parse(['unregister', '--root', str(root), '--inbox', str(inbox)]))
        self.assertFalse(config.exists())
        self.assertTrue((inbox / 'manual.txt').is_file())
