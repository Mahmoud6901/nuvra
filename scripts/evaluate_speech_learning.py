"""Compare the installed and trained Whisper models on the same dev or test recordings.

Example: python scripts/evaluate_speech_learning.py --manifest corpus/manifest.json --split dev
    --baseline models/large-v3-turbo-egyptian --candidate models/candidate --out dev.json

Reports contain reference and recognized text. Use a local output path. Running with just --baseline
first is supported; rerunning with the same output resumes after the last checkpoint.
"""
import argparse
from collections import Counter
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from arabic_relay import locate, models, paths, speech, speech_worker
from scripts.speech_benchmark import edit_counts, percentile, tokens


def file_hash(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load_split(manifest, split):
    base = manifest.resolve().parent
    selected, ids, recordings, sentences, speakers = [], set(), {}, {}, {}
    for clip in json.loads(manifest.read_text(encoding='utf-8-sig'))['clips']:
        clip_id, partition = clip['id'], clip['split']
        if clip_id in ids or partition not in ('train', 'dev', 'test'):
            raise ValueError(f'Duplicate ID or unknown split at {clip_id!r}')
        ids.add(clip_id)
        audio = (base / clip['path']).resolve()
        if not audio.is_relative_to(base):
            raise ValueError(f'Audio path outside corpus at {clip_id!r}')
        digest = file_hash(audio)
        if digest in recordings:
            raise ValueError(f'Duplicate recording at {clip_id!r} and {recordings[digest]!r}')
        recordings[digest] = clip_id
        sentence = re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', clip['reference']).casefold())
        if sentences.setdefault(sentence, partition) != partition:
            raise ValueError(f'Reference sentence appears in multiple splits at {clip_id!r}')
        speaker = clip.get('speaker')
        if speaker and speakers.setdefault(speaker, partition) != partition:
            raise ValueError(f'Speaker appears in multiple splits at {clip_id!r}')
        if partition == split:
            if not tokens(clip['reference']) or not clip['group']:
                raise ValueError(f'Missing reference words or group at {clip_id!r}')
            selected.append(dict(clip, audio=audio, sha256=digest))
    if not selected:
        raise ValueError(f'Empty {split} split')
    return selected


def score(reference, hypothesis):
    words = tokens(reference)
    counts = edit_counts(words, tokens(hypothesis))
    # Count English word occurrences, including those attached to Arabic articles.
    pattern = r"(?<![a-z0-9])(?:[a-z]|[0-9]+[a-z])[a-z0-9]*(?:[-'_.][a-z0-9]+)*(?:\+\+|#)?"
    expected = Counter(re.findall(pattern, reference.casefold()))
    heard = Counter(re.findall(pattern, hypothesis.casefold()))
    return {'words': len(words), 'edits': sum(counts), 'substitutions': counts[0],
            'deletions': counts[1], 'insertions': counts[2],
            'englishWords': expected.total(), 'englishKept': (expected & heard).total()}


def summaries(rows):
    groups = {}
    for row in rows:
        for name in ('all', row['group']):
            groups.setdefault((row['model'], row['mode'], name), []).append(row)
    output = []
    for (model, mode, group), items in sorted(groups.items()):
        words = sum(r['words'] for r in items)
        edits = sum(r['edits'] for r in items)
        english = sum(r['englishWords'] for r in items)
        kept = sum(r['englishKept'] for r in items)
        waits = [r['decodeMs'] for r in items]
        output.append({'model': model, 'mode': mode, 'group': group, 'clips': len(items),
                       'wordErrors': edits, 'referenceWords': words, 'wer': edits / words,
                       'englishWords': english, 'englishKept': kept, 'englishRecall': kept / english if english else None,
                       'medianDecodeMs': round(percentile(waits, .5)), 'p95DecodeMs': round(percentile(waits, .95))})
    return output


def checkpoint(out, report):
    report['summary'] = summaries(report['rows'])
    temp = out.with_suffix(out.suffix + '.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    temp.replace(out)


def paired_comparison(rows):
    """Paired by recording within each writing mode. Negative WER deltas favor the trained model."""
    by_key = {(r['mode'], r['model'], r['id']): r for r in rows}
    comparison = []
    for mode in sorted({r['mode'] for r in rows}):
        pairs = []
        for r in rows:
            if r['mode'] == mode and r['model'] == 'baseline':
                candidate = by_key.get((mode, 'candidate', r['id']))
                if candidate:
                    pairs.append((candidate['edits'] - r['edits'], r['words']))
        if not pairs:
            continue
        def delta(sample):
            return sum(edits for edits, words in sample) / sum(words for edits, words in sample)
        rng = random.Random(42)
        draws = [delta(rng.choices(pairs, k=len(pairs))) for _ in range(2000)]
        comparison.append({'mode': mode, 'pairedClips': len(pairs), 'werDelta': delta(pairs),
                           'recordingBootstrap95': [percentile(draws, .025), percentile(draws, .975)]})
    return comparison


def measure(name, folder, clips, modes, device, keywords, report, out):
    folder = locate.validate_model_folder(folder)
    identity = {p.name: file_hash(p) for p in folder.iterdir()
                if p.name in ('model.bin', 'vocabulary.json', 'vocabulary.txt', 'tokenizer.json', 'config.json',
                              'preprocessor_config.json')}
    if report['models'].setdefault(name, identity) != identity:
        raise ValueError(f'{name} weights or configuration changed; use a fresh output')
    completed = {(r['id'], r['mode']) for r in report['rows'] if r['model'] == name}
    remaining = [(clip, [mode for mode in modes if (clip['id'], mode) not in completed]) for clip in clips]
    remaining = [(clip, missing) for clip, missing in remaining if missing]
    if not remaining:
        return
    started = time.perf_counter()
    model = speech_worker.load_model(str(folder), device)
    overrides = speech.language_header('ar')['overrides']
    # Discard one warm-up decode in each mode. Duration excludes audio I/O, model load and translation.
    warmup = speech_worker.decode_audio_bytes(remaining[0][0]['audio'].read_bytes())
    for mode in modes:
        speech_worker.decode(model, warmup, device, keywords, inline_mode=(mode == 'inline'), overrides=overrides)
    report['loadAndWarmMs'][name] = round((time.perf_counter() - started) * 1000)
    try:
        for index, (clip, missing) in enumerate(remaining, 1):
            samples = speech_worker.decode_audio_bytes(clip['audio'].read_bytes())
            for mode in missing:
                start = time.perf_counter()
                result = speech_worker.decode(model, samples, device, keywords, inline_mode=(mode == 'inline'), overrides=overrides)
                ms = round(1000 * (time.perf_counter() - start), 2)
                text = result['text'].strip()
                report['rows'].append({'model': name, 'id': clip['id'], 'group': clip['group'], 'mode': mode,
                                       'reference': clip['reference'], 'text': text, 'seconds': len(samples) / 16000,
                                       'decodeMs': ms, **score(clip['reference'], text)})
            if index % 10 == 0:
                checkpoint(out, report)
                print(f'{name}: {index}/{len(remaining)} recordings', flush=True)
    finally:
        checkpoint(out, report)
        del model
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--modes', choices=('inline', 'craft'), nargs='+', default=['inline', 'craft'])
    parser.add_argument('--vocabulary', default='', help='Same typed hints for both models; e.g. JEV')
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    clips = load_split(args.manifest, args.split)
    modes = list(dict.fromkeys(args.modes))
    keywords = speech.parse_keywords(args.vocabulary)
    conditions = {'manifest': file_hash(args.manifest), 'split': args.split,
                  'audio': {c['id']: c['sha256'] for c in clips}, 'device': args.device, 'modes': modes,
                  'gpu': models.gpu_info() if args.device == 'cuda' else None, 'language': 'ar', 'keywords': keywords,
                  'worker': file_hash(Path(speech_worker.__file__)), 'benchmark': file_hash(Path(__file__)),
                  'versions': {p: importlib.metadata.version(p) for p in ('faster-whisper', 'ctranslate2')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        report = json.loads(args.out.read_text(encoding='utf-8'))
        if report['conditions'] != conditions:
            raise ValueError('Evaluation conditions changed; use a fresh output')
    else:
        report = {'conditions': conditions, 'models': {}, 'rows': [], 'loadAndWarmMs': {}}
    dll_handle = None
    if os.name == 'nt' and args.device == 'cuda':
        ctypes.windll.kernel32.SetDefaultDllDirectories(0x1000)
        runtime = models.gpu_runtime_dir() or paths.DEFAULT_DATA / 'runtime'
        if runtime.is_dir():
            dll_handle = os.add_dll_directory(str(runtime))
    try:
        measure('baseline', args.baseline, clips, modes, args.device, keywords, report, args.out)
        if args.candidate:
            measure('candidate', args.candidate, clips, modes, args.device, keywords, report, args.out)
            report['comparison'] = paired_comparison(report['rows'])
            checkpoint(args.out, report)
    finally:
        if dll_handle:
            dll_handle.close()
    for row in report['summary']:
        print(json.dumps(row), flush=True)
    for row in report.get('comparison', []):
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
