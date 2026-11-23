import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import evaluate_speech_learning as evaluation


class SpeechLearningEvaluation(unittest.TestCase):
    def test_micro_wer_is_weighted_by_words_and_english_recall_counts_occurrences(self):
        miss = evaluation.score('راجع الـAPI وAPI', 'راجع API')
        self.assertEqual((miss['englishKept'], miss['englishWords']), (1, 2))
        rows = [{'group': 'mixed', **miss},
                {'group': 'egyptian', **evaluation.score('قال', 'راح')}]
        for row in rows:
            row.update(model='baseline', mode='inline', decodeMs=120)
        summary = {s['group']: s for s in evaluation.summaries(rows)}
        self.assertEqual(summary['all']['wer'], (miss['edits'] + 1) / 4)
        self.assertIsNone(summary['egyptian']['englishRecall'])
        self.assertEqual(summary['mixed']['englishRecall'], .5)
        exact = evaluation.score('الـC++ و320neo', 'الـC و320 neo')
        self.assertEqual((exact['englishKept'], exact['englishWords']), (0, 2))

    def test_bootstrap_pairs_by_recording_and_reports_candidate_minus_baseline(self):
        rows = [{'id': 'a', 'model': 'baseline', 'edits': 2}, {'id': 'b', 'model': 'baseline', 'edits': 1},
                {'id': 'b', 'model': 'candidate', 'edits': 0}, {'id': 'a', 'model': 'candidate', 'edits': 1}]
        result = evaluation.paired_comparison([{**r, 'words': 10, 'mode': 'inline'} for r in rows])[0]
        self.assertEqual(result['werDelta'], -.1)
        self.assertEqual(result['recordingBootstrap95'], [-.1, -.1])

    def test_manifest_refuses_copied_recording_in_training_and_test(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            clips = []
            for split in ('train', 'test'):
                (root / split).write_bytes(b'the same recording')
                clips.append({'id': split, 'path': split, 'group': 'mixed', 'split': split, 'reference': 'افتح API'})
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'clips': clips}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate recording'):
                evaluation.load_split(manifest, 'test')
            (root / 'test').write_bytes(b'another recording')
            clips[1]['reference'] = 'افتح api!'
            manifest.write_text(json.dumps({'clips': clips}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Reference sentence appears in multiple splits'):
                evaluation.load_split(manifest, 'test')
            clips[1]['reference'] = 'بكرة'
            for clip in clips:
                clip['speaker'] = 'same-speaker'
            manifest.write_text(json.dumps({'clips': clips}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Speaker appears in multiple splits'):
                evaluation.load_split(manifest, 'test')

    def test_paired_decoder_settings_and_resume_with_changed_weights(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / 'baseline', root / 'candidate'
            for folder in (baseline, candidate):
                folder.mkdir()
                for name in ('config.json', 'tokenizer.json', 'model.bin', 'vocabulary.json'):
                    (folder / name).write_text(folder.name)
            path = root / 'sample.wav'
            path.write_bytes(b'synthetic')
            clips = [{'id': 'clip1', 'audio': path, 'reference': 'افتح الـAPI', 'group': 'mixed'}]
            report = {'models': {}, 'rows': [], 'loadAndWarmMs': {}}
            out = root / 'results.json'
            with patch.object(evaluation.speech_worker, 'load_model', side_effect=[1, 2]) as load, \
                    patch.object(evaluation.speech_worker, 'decode_audio_bytes', return_value=(0, 0)), \
                    patch.object(evaluation.speech_worker, 'decode', return_value={'text': 'افتح الـAPI'}) as decode:
                for name, folder in (('baseline', baseline), ('candidate', candidate), ('baseline', baseline)):
                    evaluation.measure(name, folder, clips, ['inline', 'craft'], 'cpu', ['JEV'], report, out)
                self.assertEqual(load.call_count, 2)
                settings = [(args[2:], kwargs) for args, kwargs in decode.call_args_list]
                self.assertEqual(settings[:4], settings[4:])
                self.assertEqual([kwargs['inline_mode'] for _, kwargs in settings[:4]], [True, False, True, False])
                for args, kwargs in settings:
                    self.assertEqual(args, ('cpu', ['JEV']))
                    self.assertEqual(kwargs['overrides'], {'language': 'ar', 'multilingual': False})
                self.assertEqual(len(json.loads(out.read_text(encoding='utf-8'))['rows']), 4)
                (candidate / 'model.bin').write_bytes(b'different weights')
                with self.assertRaisesRegex(ValueError, 'weights or configuration changed'):
                    evaluation.measure('candidate', candidate, clips, ['inline'], 'cpu', ['JEV'], report, out)
