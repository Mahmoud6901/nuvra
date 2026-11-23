import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from arabic_relay import paths, speech_learning as learning
from scripts import retrain_when_idle as idle


class LocalDictationCapture(TestCase):
    audio = bytes.fromhex('1a45dfa3') + b'simulated local dictation'
    heard = 'ما تفتحش الملف لسه'

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inbox = self.root / 'training inbox'
        self.inbox.mkdir()
        self.verify = self.model('large-v3', 32)
        self.foreground = self.model('large-v3-turbo-egyptian', 18)
        self.stem = 'nuvra-' + hashlib.sha256(self.audio).hexdigest()
        self.config = self.root / 'speech-learning.json'
        self.enterContext(patch.object(paths, 'DATA', self.root))
        self.enterContext(patch.dict(learning.RECORDINGS, clear=True))
        self.enterContext(patch.object(learning, 'LAST', None))
        self.enterContext(patch.object(learning, 'LATEST', None))
        self.enterContext(patch.object(learning.threading, 'Timer'))
        self.enterContext(patch.object(learning.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**11)))

    def model(self, folder, size):
        model = self.root / folder
        model.mkdir()
        (model / 'model.bin').write_bytes(bytes(size))
        return model

    def configure(self, automatic=True):
        self.config.write_text(json.dumps({'inbox': str(self.inbox), 'automaticLabels': automatic,
                                           'verifyModel': str(self.verify)}), encoding='utf-8')

    def offer(self, output=None, model=None, seconds=2):
        return learning.offer({'rawText': self.heard, 'text': output or self.heard},
                              model or self.foreground, seconds=seconds, audio=self.audio)

    def keep(self, response, audio=None):
        return learning.keep(response['learning']['id'], self.audio if audio is None else audio, 'audio/webm;codecs=opus')

    def correct(self, response, text):
        return learning.correct(response['learning']['id'], text)['saved']

    def test_absent_malformed_or_relative_settings_never_offer_audio_retention(self):
        for setting in (None, '{', '[]', '{"automaticLabels":"false"}',
                        json.dumps({'automaticLabels': True, 'inbox': 'relative'})):
            with self.subTest(setting=setting):
                if setting is not None:
                    self.config.write_text(setting, encoding='utf-8')
                self.assertNotIn('learning', self.offer())
                self.assertIsNone(learning.mode())
                self.assertEqual(list(self.inbox.iterdir()), [])

    def test_machine_proposal_uses_raw_words_and_publishes_nothing_before_verification(self):
        self.configure()
        response = self.offer(output='ما تفتحش ملف JEV لسه')
        self.assertTrue(self.keep(response)['staged'])
        self.assertFalse(self.keep(response)['staged'])
        self.assertEqual(learning.last()['text'], self.heard)
        self.assertEqual(list(self.inbox.glob('*.txt')), [])
        pending = self.inbox / '.pending'
        self.assertEqual((pending / (self.stem + '.txt')).read_text(encoding='utf-8'), self.heard)
        self.assertEqual((pending / (self.stem + '.webm')).read_bytes(), self.audio)

    def test_manual_only_keeps_only_a_corrected_source_and_still_checks_original_audio(self):
        self.configure(False)
        response = self.offer()
        self.assertEqual(self.keep(response), {'kept': True, 'staged': False})
        self.assertFalse(self.correct(response, self.heard))
        with self.assertRaises(ValueError):
            self.keep(response, self.audio + b'other')
        self.assertEqual(list(self.inbox.iterdir()), [])
        self.assertTrue(self.correct(response, 'راجع تقرير المصروفات بكرة الصبح'))
        self.assertEqual((self.inbox / (self.stem + '.txt')).read_text(encoding='utf-8'), 'راجع تقرير المصروفات بكرة الصبح')
        self.assertEqual((self.inbox / (self.stem + '.webm')).read_bytes(), self.audio)

    def test_same_or_unidentifiable_recognizer_requires_human_correction(self):
        self.configure()
        for model in (self.verify, self.model('copy-large-v3', 18),
                      self.model('same-binary-large-v3-turbo', 32), self.model('unknown', 21)):
            with self.subTest(model=model):
                response = self.offer(model=model)
                self.assertFalse(self.keep(response)['staged'])
        self.assertEqual(list(self.inbox.iterdir()), [])

    def test_changing_configuration_before_save_is_honored(self):
        self.configure()
        response = self.offer()
        self.config.unlink()
        self.assertFalse(self.keep(response)['kept'])
        with self.assertRaises(ValueError):
            self.correct(response, 'ما تفتحش المجلد لسه')
        self.assertIsNone(learning.last()['id'])
        self.assertEqual(list(self.inbox.iterdir()), [])

    def test_human_update_replaces_pending_and_repeat_proposal_cannot_overwrite_it(self):
        self.configure()
        response = self.offer()
        self.assertTrue(self.keep(response)['staged'])
        self.assertTrue(self.correct(response, 'ما تفتحش المجلد لسه'))
        self.assertFalse(self.keep(response)['staged'])
        self.assertEqual(list((self.inbox / '.pending').iterdir()), [])
        self.assertEqual((self.inbox / (self.stem + '.txt')).read_text(encoding='utf-8'), 'ما تفتحش المجلد لسه')
        self.assertEqual((self.inbox / (self.stem + '.webm')).read_bytes(), self.audio)

    def test_inbox_written_by_the_app_is_accepted_by_exactly_its_independent_verifier(self):
        self.configure()
        response = self.offer()
        self.assertTrue(self.keep(response)['staged'])
        with patch.object(idle, 'audio_seconds', return_value=2), patch.object(idle, 'load_verifier') as load, \
                patch.object(idle, 'verify_transcript', return_value=self.heard):
            self.assertEqual(idle.verify_pending(self.inbox, self.verify), (1, 0))
        load.assert_called_once_with(self.verify)
        self.assertEqual((self.inbox / (self.stem + '.txt')).read_text(encoding='utf-8'), self.heard)
        self.assertEqual(list((self.inbox / '.pending').iterdir()), [])

    def test_expired_recording_refuses_a_late_correction(self):
        self.configure()
        self.assertNotIn('learning', self.offer(seconds=31))
        self.configure(False)
        with patch.object(learning.time, 'monotonic', return_value=1):
            response = self.offer()
            self.keep(response)
        with patch.object(learning.time, 'monotonic', return_value=learning.KEPT_SECONDS + 2):
            self.assertIsNone(learning.last()['id'])
            with self.assertRaises(ValueError):
                self.correct(response, 'كلام مختلف')
        self.assertEqual(list(self.inbox.iterdir()), [])

    def test_late_audio_after_a_newer_ineligible_recording_is_not_offered_as_last(self):
        self.configure(False)
        earlier = self.offer()
        newer = self.offer(seconds=42)
        self.assertNotIn('learning', newer)
        self.keep(earlier)
        self.assertIsNone(learning.last()['id'])
