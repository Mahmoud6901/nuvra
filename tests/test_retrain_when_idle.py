import contextlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import retrain_when_idle as idle
from arabic_relay import paths, store


class IdleInbox(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = idle.parse(['run', '--root', str(self.root)])
        self.args.inbox.mkdir()
        public = self.args.manifest.parent
        public.mkdir()
        (public / 'test.wav').write_bytes(b'public test recording')
        self.args.manifest.write_text(json.dumps({'clips': [{'id': 'test', 'path': 'test.wav', 'split': 'test',
                                                              'reference': 'دي جملة للاختبار'}]}), encoding='utf8')
        self.now = time.time() + 2 * idle.SETTLE_SECONDS + 1
        self.stack = self.enterContext(contextlib.ExitStack())
        self.stack.enter_context(patch.object(idle, 'audio_seconds', return_value=2.0))
        self.blocked = self.stack.enter_context(patch.object(idle, 'blocked', return_value=None))
        self.stack.enter_context(patch.object(idle, 'Watchdog', return_value=contextlib.nullcontext()))
        self.train = self.stack.enter_context(patch.object(idle, 'run_training', side_effect=self.train_fake))
        profile = self.root / 'nuvra profile'
        profile.mkdir()
        self.stack.enter_context(patch.object(paths, 'DATA', profile))
        store.set_setting('whisper_model_dir', 'user-selected-before-training')

    def train_fake(self, command, log):
        self.assertIn('--if-changed', command)
        self.assertIn(str(self.args.manifest), command)
        model = self.root / 'models' / 'trained'
        model.mkdir(parents=True, exist_ok=True)
        (model / 'model.bin').write_bytes(b'test')
        report = f'runs/{self.train.call_count}.json'
        idle.train_speech.write_json(self.root / report, {'model': model.name, 'accepted': True})
        idle.train_speech.write_json(self.root / 'state.json', {'model': model.name, 'lastRun': report})
        return 0

    def put(self, file, data):
        path = self.args.inbox / file
        path.write_bytes(data.encode('utf-8') if isinstance(data, str) else data)
        return path

    def check(self):
        return idle.check(self.args, now=self.now)

    def report(self):
        return idle.Status(self.root / 'idle').value

    def test_empty_unpaired_and_unsettled_inputs_never_train(self):
        self.assertEqual(self.check(), 0)
        self.put('sound.wav', b'a')
        self.put('other.txt', 'النص')
        self.assertEqual(self.check(), 0)
        self.assertEqual(len(self.report()['inbox']['ignored']), 2)
        self.put('sound.txt', 'الصوت')
        self.now = time.time()
        self.assertEqual(self.check(), 0)
        self.train.assert_not_called()

    def test_active_meeting_blocks_idle_verification_and_training_even_during_a_manual_check(self):
        self.args.now = True
        self.put('waiting.wav', b'manually approved audio')
        self.put('waiting.txt', 'احفظ الملف')
        with patch.object(idle.meetings, 'active', return_value=True), \
                patch.object(idle, 'meeting_work') as search:
            self.assertEqual(self.check(), 0)
            search.assert_not_called()
        self.assertIn('meeting', self.report()['message'])
        self.train.assert_not_called()

    def test_new_pair_waits_for_idle_changed_label_retrains_and_model_selection_survives(self):
        self.put('note.wav', b'personal audio')
        self.put('note.txt', 'افتح الملف')
        self.blocked.return_value = 'waiting for the user'
        self.assertEqual(self.check(), 0)
        self.train.assert_not_called()
        self.blocked.return_value = None
        self.assertEqual(self.check(), 0)
        self.train.assert_called_once()
        self.assertEqual(self.report()['latestModel'], str(self.root / 'models' / 'trained'))
        manifest = self.args.inbox / '.nuvra-manifest.json'
        first, = json.loads(manifest.read_text(encoding='utf8'))['clips']
        self.assertEqual((first['split'], first['reference']), ('train', 'افتح الملف'))
        self.assertEqual(self.check(), 0)
        self.train.assert_called_once()
        (self.args.inbox / 'note.wav').rename(self.args.inbox / 'renamed.wav')
        (self.args.inbox / 'note.txt').rename(self.args.inbox / 'renamed.txt')
        self.assertEqual(self.check(), 0)
        self.train.assert_called_once()
        self.put('renamed.txt', 'افتح الملف بعد الاجتماع')
        self.assertEqual(self.check(), 0)
        self.assertEqual(self.train.call_count, 2)
        changed, = json.loads(manifest.read_text(encoding='utf8'))['clips']
        self.assertEqual(changed['id'], first['id'])
        self.assertEqual(changed['reference'], 'افتح الملف بعد الاجتماع')
        self.assertEqual(store.get_setting('whisper_model_dir'), 'user-selected-before-training')

    def test_trainer_noop_is_never_reported_as_a_new_model(self):
        self.put('repeat.wav', b'already used')
        self.put('repeat.txt', 'نفس الكلام')
        self.train.side_effect = None
        self.train.return_value = 0
        self.assertEqual(self.check(), 0)
        self.assertFalse(self.report().get('latestModel'))
        self.assertNotIn('a new model is ready', self.report()['message'])

    def test_pending_machine_labels_are_verified_only_while_idle_and_enabled(self):
        self.args.verify_model = self.root / 'full-whisper-model'
        (self.args.inbox / '.pending').mkdir()
        self.put('.pending/note.wav', b'personal recording')
        self.put('.pending/note.txt', 'ما تشغلش الملف')
        with patch.object(idle, 'load_verifier') as load, \
                patch.object(idle, 'verify_transcript', return_value='شغل الملف') as decode:
            self.blocked.return_value = 'the computer is in use'
            self.assertEqual(self.check(), 0)
            load.assert_not_called()
            self.blocked.return_value = None
            verifier = self.args.verify_model
            self.args.verify_model = None
            self.assertEqual(self.check(), 0)
            load.assert_not_called()
            self.args.verify_model = verifier
            self.assertEqual(self.check(), 0)
            load.assert_called_once_with(verifier)
            decode.assert_called_once()
        self.assertEqual(self.report()['lastVerification']['disagreed'], 1)
        self.train.assert_not_called()

    def test_rejects_conflicting_personal_labels_and_copies_of_held_out_audio(self):
        self.put('a.wav', b'personal')
        self.put('a.txt', 'قال حاجة')
        self.put('b.wav', b'personal')
        self.put('b.txt', 'قال حاجات')
        self.put('c.webm', b'public test recording')
        self.put('c.txt', 'أي حاجة')
        self.assertEqual(self.check(), 0)
        self.assertEqual(self.report()['inbox']['pairs'], 0)
        self.assertEqual(len(self.report()['inbox']['ignored']), 3)
        self.train.assert_not_called()

    def test_failed_training_retries_later_and_never_marks_pair_learned(self):
        self.put('a.wav', b'a')
        self.put('a.txt', 'كلام مسموع')
        self.train.side_effect = [1, 3]
        self.assertEqual(self.check(), 1)
        self.assertFalse(self.report().get('learned'))
        self.assertEqual(self.check(), 0)
        self.train.assert_called_once()
        self.now = self.report()['retryAfter'] + 1
        self.assertEqual(self.check(), 0)
        self.assertEqual(self.train.call_count, 2)
        self.assertEqual(self.check(), 0)
        self.assertEqual(self.train.call_count, 2)
        self.assertEqual(store.get_setting('whisper_model_dir'), 'user-selected-before-training')


class IdleConditions(unittest.TestCase):
    def test_uncertain_power_status_never_starts_a_mains_only_run(self):
        kernel = SimpleNamespace(GetSystemPowerStatus=lambda info: 0)
        with patch.object(idle.ctypes, 'windll', SimpleNamespace(kernel32=kernel), create=True), \
                patch.object(idle, 'idle_seconds', return_value=3600), \
                patch.object(idle, 'gpu_state', return_value=(9000, 0)), \
                patch.object(idle.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 10**9)):
            self.assertIsNotNone(idle.blocked(Path.cwd(), 10))

    def test_user_returns_even_if_they_only_click_once_then_read(self):
        stop = Mock()
        # Training started with 620 seconds idle. A single mouse click resets the idle counter.
        with patch.object(idle, 'on_battery', return_value=False), \
                patch.object(idle, 'idle_seconds', side_effect=[620, 4, 14, 24]):
            watcher = idle.Watchdog(stop)
            with patch.object(watcher.done, 'wait', side_effect=[False, False, False, True]):
                watcher.watch()
        stop.assert_called_once()
