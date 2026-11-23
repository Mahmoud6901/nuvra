import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from scripts import train_speech as learning


class Corpus(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / 'manifest.json'
        self.rows = []
        for split, reference in (('train', 'راجع التقرير النهاردة'), ('dev', 'بكرة نشوف الأخبار'), ('test', 'فين الحاجات دي')):
            name = split + '.wav'
            (self.root / name).write_bytes(split.encode())
            self.rows.append(dict(id=split, path=name, split=split, group='egyptian', reference=reference))

    def load(self):
        self.manifest.write_text(json.dumps({'clips': self.rows}), encoding='utf8')
        return learning.load_corpus([self.manifest])

    def test_trains_on_approved_rows_once_and_never_encodes_test(self):
        self.rows.extend([dict(self.rows[0], id='duplicate'), {'approved': False}])
        corpus, counts = self.load()
        self.assertEqual({part: [c['id'] for c in rows] for part, rows in corpus.items()},
                         {'train': ['train'], 'dev': ['dev']})
        self.assertEqual((counts['testExcluded'], counts['duplicates'], counts['unapproved']), (1, 1, 1))
        self.assertEqual(corpus['train'][0]['language'], 'ar')
        before = learning.fingerprint(corpus, {}, 'base')
        self.rows[0]['reference'] += ' لو سمحت'
        self.assertNotEqual(before, learning.fingerprint(self.load()[0], {}, 'base'))

    def test_test_audio_with_false_declared_digest_cannot_hide_training_overlap(self):
        self.rows[-1]['audio_sha256'] = 'f' * 64
        (self.root / 'test.wav').write_bytes((self.root / 'train.wav').read_bytes())
        with self.assertRaises(learning.InputError):
            self.load()

    def test_missing_test_file_is_rejected_even_with_a_declared_digest(self):
        self.rows[-1]['audio_sha256'] = learning.sha256_file(self.root / 'test.wav')
        (self.root / 'test.wav').unlink()
        with self.assertRaises(learning.InputError):
            self.load()

    def test_disjoint_audio_also_checks_source_recording_id_and_repeated_reference(self):
        self.rows[0]['recording'] = self.rows[-1]['recording'] = 'source-23'
        with self.assertRaisesRegex(learning.InputError, 'recording'):
            self.load()
        self.rows[-1]['recording'] = 'source-45'
        self.rows[-1]['reference'] = self.rows[0]['reference'] + '؟'
        with self.assertRaisesRegex(learning.InputError, 'sentence'):
            self.load()

    def test_checkpoint_identity_includes_preprocessing_and_tokenizer(self):
        for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'preprocessor_config.json', 'model.safetensors'):
            (self.root / name).write_bytes(b'{}')
        before = learning.base_inputs_hash(self.root)
        for name in ('tokenizer.json', 'preprocessor_config.json', 'generation_config.json', 'config.json'):
            with self.subTest(name=name):
                path = self.root / name
                path.write_bytes(b'{"changed": true}')
                self.assertNotEqual(before, learning.base_inputs_hash(self.root))
                path.write_bytes(b'{}')

    def test_simultaneous_if_changed_runs_do_not_train_or_update_state_twice(self):
        self.load()
        base = self.root / 'base'
        base.mkdir()
        for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'preprocessor_config.json', 'model.safetensors'):
            (base / name).write_text('{}')
        probe = """
import os, pathlib, sys, time
from scripts import train_speech as t
root, role = pathlib.Path(sys.argv[1]), sys.argv[2]
def slow_train(args, corpus, counts, base_hash, fingerprint):
    (root / ('training-' + role)).touch()
    deadline = time.monotonic() + 8
    while not (root / 'release').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    t.write_json(root / 'state.json', {'fingerprint': fingerprint})
    return 0
t.train = slow_train
(root / ('attempt-' + role)).touch()
raise SystemExit(t.main(['--base', str(root/'base'), '--root', str(root), '--manifest', str(root/'manifest.json'), '--if-changed']))
"""
        def launch(role):
            return subprocess.Popen([sys.executable, '-c', probe, str(self.root), role], cwd=learning.ROOT,
                                    env={**os.environ, 'PYTHONUTF8': '1'}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def wait_until(exists):
            deadline = time.monotonic() + 5
            while not exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(exists())

        children = []
        try:
            first = launch('first')
            children.append(first)
            wait_until((self.root / 'training-first').exists)
            second = launch('second')
            children.append(second)
            wait_until((self.root / 'attempt-second').exists)
            # A blocking lock may wait; a nonblocking lock may exit immediately.
            try:
                second.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
            (self.root / 'release').touch()
            self.assertEqual(first.wait(timeout=10), 0)
            second.wait(timeout=10)
            self.assertFalse((self.root / 'training-second').exists())
            self.assertIn('fingerprint', learning.read_state(self.root))
            repeat = launch('repeat')
            children.append(repeat)
            self.assertEqual(repeat.wait(timeout=10), 0)
            self.assertFalse((self.root / 'training-repeat').exists())
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()


class Training(unittest.TestCase):
    def test_better_average_alone_cannot_hide_group_regression(self):
        baseline = {'wer': .3, 'groups': {'egyptian': .3, 'mixed': .3}}
        candidate = {'wer': .25, 'groups': {'egyptian': .31, 'mixed': .19}}
        self.assertFalse(learning.improves(candidate, baseline, tolerance=.005))
        self.assertFalse(learning.improves(baseline, baseline, tolerance=0))

    def test_timestamp_training_predicts_text_and_end_at_the_correct_offsets(self):
        vocab = learning.Vocabulary.__new__(learning.Vocabulary)
        vocab.languages, vocab.plain, vocab.sot, vocab.transcribe = {'ar': 8}, 14, 9, 11
        vocab.first_timestamp, vocab.eot = 15, 7
        clip = {'language': 'ar', 'reference': 'افتح', 'seconds': 1.0}
        with patch.object(vocab, 'encode', return_value=[1, 2]):
            self.assertEqual(vocab.target(clip, False), ([9, 8, 11, 14, 1, 2], [learning.IGNORE] * 3 + [1, 2, 7]))
            self.assertEqual(vocab.target(clip, True), ([9, 8, 11, 15, 1, 2, 65],
                                                       [learning.IGNORE] * 3 + [1, 2, learning.ANY_TIMESTAMP, 7]))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'the optional training environment supplies torch')
    def test_cpu_adapter_learns_from_gradient_and_merges_without_changing_predictions(self):
        import torch
        torch.manual_seed(7)
        decoder = torch.nn.Module()
        decoder.self_attn = torch.nn.Module()
        decoder.self_attn.q_proj = torch.nn.Linear(4, 3).requires_grad_(False)
        x = torch.tensor([[1., 2., -1., 3.]])
        original = decoder.self_attn.q_proj(x).detach().clone()
        adapter = learning.Adapter(torch, decoder, 2, 4, ['q_proj'])
        trained = lambda: decoder.self_attn.q_proj(x)
        self.assertTrue(torch.equal(trained(), original))
        loss = (trained() - 1).square().mean()
        optimizer = torch.optim.SGD(adapter.parameters(), lr=.1)
        loss.backward()
        optimizer.step()
        expected = trained().detach().clone()
        self.assertFalse(torch.allclose(expected, original))
        state = adapter.state()
        adapter.remove()
        self.assertTrue(torch.equal(trained(), original))
        restored = learning.Adapter(torch, decoder, 2, 4, ['q_proj'])
        restored.load(state)
        restored.remove(merge=True)
        torch.testing.assert_close(trained(), expected)
