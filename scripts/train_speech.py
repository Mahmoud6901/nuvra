"""Supervised fine-tuning of Nuvra's Whisper model on approved, labeled recordings.

    python scripts/train_speech.py --check          # validate the manifest only, nothing is trained
    python scripts/train_speech.py                  # train, select on dev, export a model folder for Models > Browse
    python scripts/train_speech.py --if-changed     # the same, but exits at once when no label changed (schedulable)

Run it with a separate training Python (torch, transformers, safetensors, faster-whisper); Nuvra itself needs none of
these. The frozen encoder runs once per clip and its output is cached, then only low-rank (LoRA) updates of the
decoder's attention are trained, so about 2 GB of free GPU memory is enough. Every run starts from the same base
checkpoint and learns from the whole approved corpus, so repeated runs never compound an earlier run's mistakes.

Manifest: {"clips": [{"id", "split": "train|dev|test", "group", "path" (relative to the manifest), "reference"}]}.
Optional per clip: "language" (ar|en), "approved" (false: ignored), "audio_sha256", "recording" (source recording id).
Test clips are never trained on, encoded, decoded or scored here; their files and sentences are only compared against
train and dev to reject leakage. A run whose best checkpoint does not beat the base model (and the previously accepted model) on dev exports nothing.

Exit codes: 0 exported, or nothing new to learn; 2 invalid input; 3 trained, but dev did not improve; 4 another run
is active (one run per --root at a time; the system releases the lock even when a run is killed).
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from arabic_relay.incremental import normalize  # noqa: E402

LEARNING = ROOT / '.local' / 'asr-learning'
SPLITS = ('train', 'dev', 'test')
RATE = 16000
MAX_SECONDS = 30.0
MAX_TARGET = 448                     # Whisper's decoder length, prompt included
TIMESTAMP_SHARE = 0.75               # Nuvra decodes with timestamps; some examples keep the plain prompt working
IGNORE, ANY_TIMESTAMP = -100, -200   # label sentinels: no loss; loss on "some timestamp", whichever one it is
FEATURE_VERSION = 1                  # bump when the cached encoder output would change
ALGORITHM_VERSION = 1                # bump when training, selection or export would produce a different model
# Nuvra recognizes this family by its folder name (models.is_egyptian_turbo), so exports must carry it.
FAMILY = 'large-v3-turbo-egyptian'


class InputError(Exception):
    pass


class Busy(Exception):
    pass


# --- Corpus ----------------------------------------------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as data:
        while chunk := data.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def words(text):
    return [w for w in (normalize(word) for word in re.split(r'\s+', text or '')) if w]


def read_clip(clip, manifest):
    """One checked manifest row. Every file is hashed as it is on disk, test audio included (and nothing more)."""
    if not isinstance(clip, dict):
        raise InputError(f'{manifest}: every clip must be an object')
    clip_id = clip.get('id')
    where = f'{manifest}: clip {clip_id!r}'
    for key in ('id', 'split', 'group', 'path', 'reference'):
        if not isinstance(clip.get(key), str) or not clip[key].strip():
            raise InputError(f'{where}: "{key}" must be a non-empty string')
    if clip['split'] not in SPLITS:
        raise InputError(f'{where}: split must be one of {", ".join(SPLITS)}')
    if clip.get('language') not in (None, 'ar', 'en'):
        raise InputError(f'{where}: language must be ar or en')
    folder = manifest.parent.resolve()
    path = (folder / clip['path']).resolve()
    if Path(clip['path']).is_absolute() or not path.is_relative_to(folder):
        raise InputError(f'{where}: path must stay inside the manifest folder')
    declared = clip.get('audio_sha256')
    if not path.is_file():
        raise InputError(f'{where}: audio file is missing')
    digest = sha256_file(path)
    if declared is not None and declared != digest:
        raise InputError(f'{where}: audio does not match audio_sha256')
    reference = re.sub(r'\s+', ' ', clip['reference']).strip()
    if not words(reference):
        raise InputError(f'{where}: the reference has no words')
    language = clip.get('language') or ('ar' if re.search('[ء-ي]', reference) else 'en')
    return {'id': clip['id'], 'split': clip['split'], 'group': clip['group'], 'path': path, 'sha256': digest,
            'reference': reference, 'language': language, 'recording': clip.get('recording')}


def check_leakage(clips):
    """Nothing identifying a recording or its sentence may appear in two splits."""
    seen, problems = {}, []
    for clip in clips:
        identities = [('audio file', os.path.normcase(clip['path'])), ('audio content', clip['sha256']), ('recording', clip['recording'])]
        sentence = words(clip['reference'])
        if len(sentence) >= 3:       # one- and two-word answers repeat naturally between unrelated recordings
            identities.append(('sentence', ' '.join(sentence)))
        for kind, value in identities:
            if not value:
                continue
            first = seen.setdefault((kind, value), clip)
            if first['split'] != clip['split']:
                problems.append(f'same {kind} in {first["split"]} ({first["id"]}) and {clip["split"]} ({clip["id"]})')
    if problems:
        raise InputError('leakage between splits:\n  ' + '\n  '.join(problems[:20])
                         + (f'\n  ... and {len(problems) - 20} more' if len(problems) > 20 else ''))


def load_corpus(manifests):
    """Checked train and dev clips from all manifests, plus counts. Test clips are dropped after the leakage check."""
    clips, ids, unapproved = [], set(), 0
    for manifest in (Path(m) for m in manifests):
        try:
            rows = json.loads(manifest.read_text(encoding='utf-8-sig'))['clips']
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise InputError(f'cannot read {manifest}: {error}') from None
        if not isinstance(rows, list):
            raise InputError(f'{manifest}: "clips" must be a list')
        for row in rows:
            # An unapproved test clip still guards against leakage.
            if isinstance(row, dict) and row.get('approved', True) is not True and row.get('split') != 'test':
                unapproved += 1
                continue
            clip = read_clip(row, manifest)
            if clip['id'] in ids:
                raise InputError(f'duplicate clip id {clip["id"]!r}')
            ids.add(clip['id'])
            clips.append(clip)
    check_leakage(clips)
    corpus, duplicates, seen = {'train': [], 'dev': []}, 0, set()
    for clip in clips:
        if clip['split'] == 'test':
            continue
        if clip['sha256'] in seen:
            duplicates += 1
            continue
        seen.add(clip['sha256'])
        corpus[clip['split']].append(clip)
    for split, rows in corpus.items():
        if not rows:
            raise InputError(f'the manifest has no {split} clips')
    counts = {'unapproved': unapproved, 'duplicates': duplicates,
              'testExcluded': sum(c['split'] == 'test' for c in clips)}
    for split, rows in corpus.items():
        counts[split] = {group: sum(c['group'] == group for c in rows) for group in sorted({c['group'] for c in rows})}
    return corpus, counts


def library_versions():
    versions = {}
    for name in ('torch', 'transformers', 'faster-whisper', 'ctranslate2', 'tokenizers'):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return versions


def fingerprint(corpus, settings, base_hash):
    """Everything a run learns from and with. Unchanged means a rerun could only repeat the previous result."""
    rows = sorted([c['split'], c['id'], c['sha256'], c['reference'], c['group'], c['language']]
                  for part in corpus.values() for c in part)
    return hashlib.sha256(json.dumps(
        [ALGORITHM_VERSION, FEATURE_VERSION, rows, settings, base_hash, library_versions()], sort_keys=True).encode()).hexdigest()


# --- Scoring ---------------------------------------------------------------------------------------------------

def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def score(clips, texts):
    """Word error rate per group; `wer` is the mean of the groups, so a small group is never outvoted."""
    totals = {}
    for clip, text in zip(clips, texts):
        reference = words(clip['reference'])
        group = totals.setdefault(clip['group'], [0, 0])
        group[0] += edit_distance(reference, words(text))
        group[1] += len(reference)
    groups = {name: errors / count for name, (errors, count) in sorted(totals.items())}
    return {'wer': sum(groups.values()) / len(groups), 'groups': groups}


def improves(candidate, reference, tolerance):
    """Better overall, and no group pays for it by more than `tolerance`."""
    return candidate['wer'] < reference['wer'] and all(
        candidate['groups'][group] <= value + tolerance for group, value in reference['groups'].items())


def shown(result):
    return f"WER {result['wer']:.4f} (" + ', '.join(f'{g} {v:.4f}' for g, v in result['groups'].items()) + ')'


# --- Files -----------------------------------------------------------------------------------------------------

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    temporary.replace(path)


def read_state(root):
    try:
        state = json.loads((root / 'state.json').read_text(encoding='utf-8'))
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def exclusive(root):
    """One run per root: state, cache and models. The system drops the lock when the process ends, however it ends."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'train.lock').open('a+b') as handle:
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Busy(f'another training run is using {root}') from None
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def publish(temporary, target):
    """Make the finished folder visible under its final name in one step; never replaces an existing one."""
    for attempt in range(6):
        if target.exists():
            raise InputError(f'{target} already exists')
        try:
            temporary.rename(target)
            return
        except PermissionError:      # a scanner can hold a new file open briefly on Windows
            if attempt == 5:
                raise
            time.sleep(2 ** attempt / 2)


# --- Model -----------------------------------------------------------------------------------------------------

def base_inputs_hash(base):
    """Identity of every file the model, tokenizer, features or decoding settings are read from."""
    weights = sorted(base.glob('*.safetensors'))
    needed = [base / name for name in ('config.json', 'generation_config.json', 'tokenizer.json')]
    if not weights or not all(path.is_file() for path in needed):
        raise InputError(f'{base} is not a Hugging Face Whisper checkpoint (config.json, generation_config.json, '
                         'tokenizer.json, *.safetensors)')
    inputs = [path for path in base.iterdir() if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt')]
    return hashlib.sha256(json.dumps(sorted([path.name, sha256_file(path)] for path in inputs)).encode()).hexdigest()


def preprocessor(base):
    """The feature extractor settings; newer Transformers keeps them inside processor_config.json."""
    for name, pick in (('preprocessor_config.json', lambda c: c), ('processor_config.json', lambda c: c['feature_extractor'])):
        try:
            return pick(json.loads((base / name).read_text(encoding='utf-8')))
        except (OSError, ValueError, KeyError):
            continue
    raise InputError(f'{base} has neither preprocessor_config.json nor processor_config.json')


class Vocabulary:
    """Token ids straight from tokenizer.json, as faster-whisper reads them."""

    def __init__(self, base, generation):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(base / 'tokenizer.json'))
        self.eot, self.sot = generation.eos_token_id, generation.decoder_start_token_id
        if isinstance(self.eot, list):
            self.eot = self.eot[0]
        self.languages = {name: generation.lang_to_id[f'<|{name}|>'] for name in ('ar', 'en')}
        self.transcribe = generation.task_to_id['transcribe']
        self.plain = generation.no_timestamps_token_id
        self.first_timestamp = self.plain + 1
        self.suppress = list(generation.suppress_tokens or []) + [self.plain]
        self.suppress_first = list(generation.begin_suppress_tokens or [])

    def encode(self, text):
        return self.tokenizer.encode(' ' + text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode([i for i in ids if i < self.eot])

    def target(self, clip, timestamps):
        """Decoder inputs and their labels. The language is Nuvra's own choice per segment, so it is not trained."""
        text = self.encode(clip['reference'])
        prompt = [self.sot, self.languages[clip['language']], self.transcribe]
        if timestamps:
            # A clip's duration only approximates where speech ends: teach "a timestamp here", not which one.
            end = self.first_timestamp + round(min(clip['seconds'], MAX_SECONDS) / 0.02)
            tokens = prompt + [self.first_timestamp] + text + [end, self.eot]
            labels = [IGNORE] * 3 + text + [ANY_TIMESTAMP, self.eot]
        else:
            tokens = prompt + [self.plain] + text + [self.eot]
            labels = [IGNORE] * 3 + text + [self.eot]
        return tokens[:-1], labels

    def tokens(self, size):
        known = self.tokenizer.get_vocab()
        names = sorted(known, key=known.get)
        if [known[name] for name in names] != list(range(len(names))) or len(names) > size:
            raise InputError('tokenizer.json does not match the model vocabulary')
        return names + [f'<|{i * 0.02:.2f}|>' for i in range(size - len(names))]


def load_model(base, dtype):
    from transformers import WhisperForConditionalGeneration
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=dtype, local_files_only=True,
                                                            use_safetensors=True)
    model.eval().requires_grad_(False)
    return model


def lora_layers(decoder, targets):
    """(parent, attribute, name) of the decoder attention projections to adapt."""
    from torch import nn
    found = []
    for name, module in decoder.named_modules():
        if name.rsplit('.', 1)[-1] in ('self_attn', 'encoder_attn'):
            found += [(module, target, f'{name}.{target}') for target in targets
                      if isinstance(getattr(module, target, None), nn.Linear)]
    if not found:
        raise InputError(f'no decoder attention layer matches --targets {",".join(targets)}')
    return found


def make_lora(torch, linear, rank, alpha, dropout):
    class LoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = linear
            device = linear.weight.device
            self.a = torch.nn.Parameter(torch.empty(rank, linear.in_features, device=device, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.zeros(linear.out_features, rank, device=device, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
            self.dropout = torch.nn.Dropout(dropout)
            self.scale = alpha / rank

        def forward(self, x):
            low = torch.nn.functional.linear(self.dropout(x.float()), self.a)
            update = torch.nn.functional.linear(low, self.b) * self.scale
            return self.base(x) + update.to(x.dtype)

        def merge(self):
            delta = (self.b @ self.a) * self.scale
            self.base.weight.data = (self.base.weight.data.float() + delta).to(self.base.weight.dtype)
    return LoRA()


class Adapter:
    """The trainable low-rank update of a frozen decoder; `remove` restores the untouched base."""

    def __init__(self, torch, decoder, rank, alpha, targets, dropout=0.0):
        self.settings = {'rank': rank, 'alpha': alpha, 'targets': list(targets)}
        self.slots = [(parent, attribute, name, make_lora(torch, getattr(parent, attribute), rank, alpha, dropout))
                      for parent, attribute, name in lora_layers(decoder, targets)]
        for parent, attribute, _, layer in self.slots:
            setattr(parent, attribute, layer)

    def parameters(self):
        return [p for *_, layer in self.slots for p in (layer.a, layer.b)]

    def state(self):
        return {f'{name}.{part}': getattr(layer, part).detach().cpu().clone().contiguous()
                for _, _, name, layer in self.slots for part in ('a', 'b')}

    def load(self, state):
        for _, _, name, layer in self.slots:
            for part in ('a', 'b'):
                getattr(layer, part).data.copy_(state[f'{name}.{part}'])

    def remove(self, merge=False):
        for parent, attribute, _, layer in self.slots:
            if merge:
                layer.merge()
            setattr(parent, attribute, layer.base)


# --- Encoder cache ---------------------------------------------------------------------------------------------

class Features:
    """The frozen encoder's output per clip, on disk (about 4 MB each): computed once, reused by every later run."""

    def __init__(self, folder, base_hash, device):
        versions = {name: version for name, version in library_versions().items() if name in ('transformers', 'faster-whisper')}
        identity = json.dumps([FEATURE_VERSION, base_hash, device, versions], sort_keys=True)
        self.folder = folder / hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, clip):
        return self.folder / (clip['sha256'] + '.npy')

    def cached(self, clip, shape):
        import numpy as np
        try:
            return np.load(self.path(clip), mmap_mode='r').shape == shape
        except (OSError, ValueError, EOFError):
            return False

    def load(self, torch, clips, device):
        import numpy as np
        return torch.from_numpy(np.stack([np.load(self.path(c)) for c in clips])).to(device)

    def fill(self, torch, model, clips, settings, device):
        """Encode what is missing; returns the clips that can be used, each with its length in seconds."""
        import numpy as np
        from faster_whisper.audio import decode_audio, pad_or_trim
        from faster_whisper.feature_extractor import FeatureExtractor
        # The same features Nuvra decodes from (faster-whisper), not the slightly different Transformers padding.
        extract = FeatureExtractor(**{k: settings[k] for k in ('feature_size', 'sampling_rate', 'hop_length',
                                                              'chunk_length', 'n_fft') if k in settings})
        encoder, usable, skipped, started, encoded = model.model.encoder, [], [], time.monotonic(), 0
        shape = (model.config.max_source_positions, model.config.d_model)
        for index, clip in enumerate(clips, 1):
            audio = clip['path'].read_bytes()
            if hashlib.sha256(audio).hexdigest() != clip['sha256']:
                raise InputError(f"{clip['id']}: the audio changed after it was checked")
            samples = decode_audio(io.BytesIO(audio), sampling_rate=RATE)
            clip['seconds'] = len(samples) / RATE
            if not 0.2 <= clip['seconds'] <= MAX_SECONDS:
                skipped.append({'id': clip['id'], 'why': f"{clip['seconds']:.1f} seconds long"})
                continue
            usable.append(clip)
            if self.cached(clip, shape):
                continue
            if not encoded:
                encoder.to(device)
            mel = pad_or_trim(extract(samples)[:, :-1])
            with torch.no_grad():
                state = encoder(torch.from_numpy(mel)[None].to(device, encoder.conv1.weight.dtype)).last_hidden_state[0]
            if not torch.isfinite(state).all():
                raise RuntimeError(f"the encoder produced invalid numbers for {clip['id']}")
            temporary = self.path(clip).with_suffix('.tmp')
            with temporary.open('wb') as out:
                np.save(out, state.to(torch.float16).cpu().numpy())
            temporary.replace(self.path(clip))
            encoded += 1
            if encoded % 100 == 0:
                print(f'  encoded {encoded} clips ({index}/{len(clips)} checked, {time.monotonic() - started:.0f} s)', flush=True)
        encoder.to('cpu')
        if device == 'cuda':
            torch.cuda.empty_cache()
        print(f'Encoder output: {encoded} computed, {len(usable) - encoded} from the cache, {len(skipped)} clips skipped', flush=True)
        return usable, skipped


# --- Training --------------------------------------------------------------------------------------------------

class Trainer:
    def __init__(self, torch, model, vocabulary, features, device, seed):
        self.torch, self.model, self.vocabulary, self.features, self.device = torch, model, vocabulary, features, device
        self.decoder = model.model.decoder
        self.random = random.Random(seed)

    def autocast(self):
        return self.torch.autocast('cuda', dtype=self.torch.float16, enabled=self.device == 'cuda')

    def states(self, clips):
        return self.features.load(self.torch, clips, self.device).to(next(self.decoder.parameters()).dtype)

    def loss_sum(self, batch):
        """Summed token loss of [(clip, inputs, labels)]."""
        torch, width = self.torch, max(len(inputs) for _, inputs, _ in batch)
        inputs = torch.tensor([x + [self.vocabulary.eot] * (width - len(x)) for _, x, _ in batch], device=self.device)
        labels = torch.tensor([y + [IGNORE] * (width - len(y)) for _, _, y in batch], device=self.device)
        with self.autocast():
            hidden = self.decoder(input_ids=inputs, encoder_hidden_states=self.states([c for c, _, _ in batch]),
                                  use_cache=False).last_hidden_state
            kept = labels != IGNORE
            logits = self.model.proj_out(hidden[kept])      # logits only where there is something to learn
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        labels = labels[kept]
        timestamp = labels == ANY_TIMESTAMP
        exact = log_probabilities.gather(1, labels.clamp(min=0)[:, None])[:, 0]
        any_timestamp = log_probabilities[:, self.vocabulary.first_timestamp:].logsumexp(dim=-1)
        return -torch.where(timestamp, any_timestamp, exact).sum()

    def backward(self, batch, scaler, micro):
        """Gradients of the mean token loss, in pieces of `micro` clips; halves the pieces when the GPU is full."""
        torch = self.torch
        count = sum(1 for _, _, labels in batch for label in labels if label != IGNORE)
        while True:
            try:
                total = 0.0
                for start in range(0, len(batch), micro):
                    loss = self.loss_sum(batch[start:start + micro]) / count
                    scaler.scale(loss).backward()
                    total += loss.item()
                return total, micro
            except torch.cuda.OutOfMemoryError:
                if micro == 1:
                    raise
                micro = max(1, micro // 2)
                self.optimizer.zero_grad(set_to_none=True)
                loss = None
                torch.cuda.empty_cache()
                print(f'  GPU memory is full: continuing with {micro} clips at a time', flush=True)

    def transcribe(self, clips, batch_size):
        """Greedy text of each clip. Starts as Nuvra's own decoding does: transcribe, at timestamp zero."""
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        torch, vocabulary, texts = self.torch, self.vocabulary, {}
        order = sorted(clips, key=lambda c: c['seconds'])
        start = 0
        while start < len(order):
            batch = order[start:start + batch_size]
            try:
                done = self._greedy(torch, vocabulary, batch, EncoderDecoderCache(DynamicCache(), DynamicCache()))
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue
            texts.update({c['id']: vocabulary.decode(ids) for c, ids in zip(batch, done)})
            start += len(batch)
        return [texts[c['id']] for c in clips]

    def _greedy(self, torch, vocabulary, batch, cache):
        ids = torch.tensor([[vocabulary.sot, vocabulary.languages[c['language']], vocabulary.transcribe,
                             vocabulary.first_timestamp] for c in batch], device=self.device)
        output = [[] for _ in batch]
        finished = torch.zeros(len(batch), dtype=torch.bool, device=self.device)
        with torch.no_grad(), self.autocast():
            states = self.states(batch)
            for step in range(MAX_TARGET - 8):
                hidden = self.decoder(input_ids=ids, encoder_hidden_states=states, past_key_values=cache, use_cache=True)
                cache = hidden.past_key_values
                logits = self.model.proj_out(hidden.last_hidden_state[:, -1]).float()
                logits[:, vocabulary.suppress + (vocabulary.suppress_first if step == 0 else [])] = -math.inf
                chosen = logits.argmax(dim=-1).masked_fill(finished, vocabulary.eot)
                for row, token in zip(output, chosen.tolist()):
                    row.append(token)
                finished |= chosen == vocabulary.eot
                if finished.all():
                    break
                ids = chosen[:, None]
        return output

    def evaluate(self, clips, batch_size):
        was_training = self.decoder.training
        self.decoder.eval()
        result = score(clips, self.transcribe(clips, batch_size))
        self.decoder.train(was_training)
        return result

    def fit(self, adapter, train, dev, baseline, args):
        """Trains the adapter; returns (best state or None, its dev result, history). Only dev decides."""
        torch = self.torch
        self.optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
        per_epoch = math.ceil(len(train) / args.batch_size)
        total = min(per_epoch * args.epochs, args.max_steps or per_epoch * args.epochs)
        warmup = max(1, total // 10)
        schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(
            (step + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(1, max(0, step - warmup) / max(1, total - warmup))))))
        scaler = torch.amp.GradScaler('cuda', enabled=self.device == 'cuda', init_scale=2.0 ** 12)
        every = args.eval_steps or per_epoch
        best_state, best, history = None, baseline, []
        step, stale, micro, losses, started = 0, 0, args.batch_size, [], time.monotonic()
        self.decoder.train()
        while step < total and stale < args.patience:
            order = train[:]
            self.random.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                batch = [(c, *self.vocabulary.target(c, self.random.random() < TIMESTAMP_SHARE))
                         for c in order[start:start + args.batch_size]]
                loss, micro = self.backward(batch, scaler, micro)
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                schedule.step()
                step += 1
                losses.append(loss)
                if len(losses) >= 5 and not any(map(math.isfinite, losses[-5:])):
                    raise RuntimeError('training diverged (the loss is not a number); lower --lr')
                if step % 20 == 0:
                    print(f'  step {step}/{total}  loss {sum(losses[-20:]) / len(losses[-20:]):.4f}  '
                          f'{time.monotonic() - started:.0f} s', flush=True)
                if step % every == 0 or step == total:
                    result = self.evaluate(dev, args.eval_batch_size)
                    better = improves(result, best, args.group_tolerance) and improves(result, baseline, args.group_tolerance)
                    history.append({'step': step, 'loss': sum(losses) / len(losses), 'dev': result, 'best': better})
                    losses = []
                    print(f'  step {step}/{total}  dev {shown(result)}' + ('  best so far' if better else ''), flush=True)
                    stale = 0 if better else stale + 1
                    if better:
                        best_state, best = adapter.state(), result
                    if stale >= args.patience or step >= total:
                        break
        self.decoder.eval()
        return best_state, best, history


# --- Export ----------------------------------------------------------------------------------------------------

def export(model, vocabulary, base, settings, folder):
    """CTranslate2 float16 model folder, as Nuvra loads it."""
    import torch
    from ctranslate2.converters.transformers import WhisperLoader
    loader = WhisperLoader()
    with torch.no_grad():
        spec = loader.get_model_spec(model)
        loader.set_config(spec.config, model, None)       # suppression and alignment from generation_config.json
        loader.set_vocabulary(spec, vocabulary.tokens(model.config.vocab_size))
        spec.validate()
        spec.optimize(quantization='float16')
        spec.save(str(folder))
    shutil.copyfile(base / 'tokenizer.json', folder / 'tokenizer.json')
    write_json(folder / 'preprocessor_config.json', settings)


def check_export(folder, clip):
    """The exported folder must load in the engine Nuvra uses and hear speech in one dev recording."""
    import gc
    from faster_whisper import WhisperModel
    missing = [n for n in ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'preprocessor_config.json')
               if not (folder / n).is_file()]
    if missing:
        raise RuntimeError('the export is missing ' + ', '.join(missing))
    engine = WhisperModel(str(folder), device='cpu', compute_type='int8', local_files_only=True)
    segments, _ = engine.transcribe(str(clip['path']), language=clip['language'], beam_size=1, word_timestamps=True)
    text = ''.join(segment.text for segment in segments).strip()
    del engine
    gc.collect()                                          # Windows cannot rename a folder with open files
    if not text:
        raise RuntimeError('the exported model returned no text for a dev recording')


# --- Run -------------------------------------------------------------------------------------------------------

def accepted_adapter(root, state, base_hash):
    """The model the last accepted run produced, when it was trained from the same base."""
    name = state.get('model')
    if not name:
        return None
    folder = root / 'models' / name
    try:
        report = json.loads((folder / 'training.json').read_text(encoding='utf-8'))
        return (folder, report['adapter']) if report['baseSha256'] == base_hash else None
    except (OSError, ValueError, KeyError):
        return None


def train(args, corpus, counts, base_hash, print_hash):
    import torch
    from safetensors.torch import load_file, save_file
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise InputError('CUDA is not available to this Python; use --device cpu')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    name = f'{FAMILY}-{args.name or stamp}'
    target = args.root / 'models' / name
    if target.exists():
        raise InputError(f'{target} already exists; choose another --name')
    state = read_state(args.root)
    for leftover in (args.root / 'models').glob(f'.{FAMILY}-*.tmp'):
        shutil.rmtree(leftover, ignore_errors=True)       # from a killed run; this run holds the lock
    torch.manual_seed(args.seed)
    settings = preprocessor(args.base)
    model = load_model(args.base, torch.float16 if args.device == 'cuda' else torch.float32)
    vocabulary = Vocabulary(args.base, model.generation_config)
    features = Features(args.root / 'features', base_hash, args.device)
    parts, skipped = {}, []
    for split, clips in corpus.items():
        fits = [c for c in clips if len(vocabulary.target({**c, 'seconds': 0}, True)[0]) <= MAX_TARGET]
        skipped += [{'id': c['id'], 'why': 'reference too long'} for c in clips if c not in fits]
        parts[split], gone = features.fill(torch, model, fits, settings, args.device)
        skipped += gone
        if not parts[split]:
            raise InputError(f'no usable {split} clips')
    model.model.decoder.to(args.device)
    model.proj_out.to(args.device)                        # shares its weights with the decoder embeddings
    trainer = Trainer(torch, model, vocabulary, features, args.device, args.seed)
    base_dev = trainer.evaluate(parts['dev'], args.eval_batch_size)
    print(f"Base model on dev: {shown(base_dev)}", flush=True)
    bar, previous = base_dev, None
    incumbent = accepted_adapter(args.root, state, base_hash)
    if incumbent:
        folder, shape = incumbent
        adapter = Adapter(torch, model.model.decoder, shape['rank'], shape['alpha'], shape['targets'])
        adapter.load(load_file(folder / 'adapter.safetensors'))
        previous = {'model': folder.name, 'dev': trainer.evaluate(parts['dev'], args.eval_batch_size)}
        adapter.remove()
        print(f"Previously accepted {folder.name} on dev: {shown(previous['dev'])}", flush=True)
    adapter = Adapter(torch, model.model.decoder, args.rank, args.alpha, args.targets, args.dropout)
    print(f"Training {sum(p.numel() for p in adapter.parameters()):,} of "
          f"{sum(p.numel() for p in model.parameters()):,} parameters on {len(parts['train'])} clips", flush=True)
    started = time.monotonic()
    best_state, best, history = trainer.fit(adapter, parts['train'], parts['dev'], bar, args)
    report = {'model': name, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'base': str(args.base), 'baseSha256': base_hash,
              'fingerprint': print_hash, 'corpus': counts, 'manifests': [str(m) for m in args.manifest], 'skipped': skipped,
              'trained': {split: len(clips) for split, clips in parts.items()}, 'adapter': adapter.settings,
              'hyperparameters': hyperparameters(args), 'seconds': round(time.monotonic() - started),
              'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
              'versions': library_versions(), 'algorithm': ALGORITHM_VERSION,
              'devBase': base_dev, 'devPrevious': previous, 'devBest': best if best_state else None, 'history': history}
    reason = None
    if best_state is None:
        reason = 'no checkpoint beat the base model on dev'
    else:
        adapter.load(best_state)
        adapter.remove(merge=True)
        # What is exported is the merged, rounded model, so that is what must still win.
        report['devMerged'] = merged = trainer.evaluate(parts['dev'], args.eval_batch_size)
        print(f'Merged model on dev: {shown(merged)}', flush=True)
        if not improves(merged, base_dev, args.group_tolerance):
            reason = 'the merged model did not beat the base model on dev'
        elif previous and not improves(merged, previous['dev'], args.group_tolerance):
            reason = f"the new model did not beat {previous['model']} on dev"
    if reason is None:
        model.to('cpu')
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        temporary = args.root / 'models' / f'.{name}.{os.getpid()}.tmp'
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            export(model, vocabulary, args.base, settings, temporary)
            save_file(best_state, temporary / 'adapter.safetensors')
            report['accepted'] = True
            write_json(temporary / 'training.json', report)
            del model, trainer
            check_export(temporary, parts['dev'][0])
            publish(temporary, target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    else:
        report.update(accepted=False, reason=reason)
    write_json(args.root / 'runs' / f'{stamp}.json', report)
    write_json(args.root / 'state.json', {**state, 'fingerprint': print_hash, 'lastRun': f'runs/{stamp}.json',
                                          **({'model': name} if reason is None else {})})
    if reason:
        print(f'Nothing exported: {reason}. Report: {args.root / "runs" / (stamp + ".json")}')
        return 3
    print(f'Model ready: {target}\nIn Nuvra choose it under Models > Browse.')
    return 0


def hyperparameters(args):
    return {key: getattr(args, key) for key in ('epochs', 'max_steps', 'batch_size', 'lr', 'rank', 'alpha', 'dropout',
                                                'targets', 'eval_steps', 'patience', 'group_tolerance', 'seed', 'limit_train')}


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', type=Path, nargs='+', default=[LEARNING / 'corpus' / 'manifest.json'],
                        help='one or more labeled manifests; all are learned from together')
    parser.add_argument('--base', type=Path, default=LEARNING / 'base', help='Hugging Face Whisper checkpoint folder')
    parser.add_argument('--root', type=Path, default=LEARNING, help='where models, reports and cached features go')
    parser.add_argument('--name', help='suffix of the exported model folder (default: the time)')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=0, help='stop after this many updates (0: no limit)')
    parser.add_argument('--batch-size', type=int, default=4, help='clips per update; split further when memory is short')
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--eval-steps', type=int, default=0, help='updates between dev evaluations (0: once per epoch)')
    parser.add_argument('--patience', type=int, default=2, help='stop after this many dev evaluations without a new best')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=32)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--targets', default='q_proj,v_proj', type=lambda v: sorted(filter(None, v.split(','))),
                        help='decoder attention projections to adapt')
    parser.add_argument('--group-tolerance', type=float, default=0.005,
                        help='how much dev WER any one group may lose while the average improves')
    parser.add_argument('--limit-train', type=int, default=0, help='use only this many random train clips (smoke test)')
    parser.add_argument('--seed', type=int, default=109)
    parser.add_argument('--if-changed', action='store_true', help='exit at once when nothing changed since the last run')
    parser.add_argument('--check', action='store_true', help='validate the manifests and stop')
    args = parser.parse_args(argv)
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'patience', 'rank'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be at least 1')
    for key in ('max_steps', 'eval_steps', 'limit_train', 'group_tolerance'):
        if getattr(args, key) < 0:
            parser.error(f'--{key.replace("_", "-")} cannot be negative')
    if not 0 < args.lr < 1 or args.alpha <= 0 or not 0 <= args.dropout < 1:
        parser.error('--lr, --alpha or --dropout is out of range')
    if args.name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', args.name):
        parser.error('--name may hold letters, digits, dots, dashes and underscores')
    for key in ('base', 'root'):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK'):
        os.environ[key] = '1'                             # recordings and labels stay on this computer
    args = parse(argv)
    try:
        with contextlib.nullcontext() if args.check else exclusive(args.root):
            corpus, counts = load_corpus(args.manifest)
            print('Corpus: ' + json.dumps(counts, ensure_ascii=False), flush=True)
            if args.check:
                return 0
            if args.limit_train:
                random.Random(args.seed).shuffle(corpus['train'])
                corpus['train'] = corpus['train'][:args.limit_train]
            base_hash = base_inputs_hash(args.base)
            print_hash = fingerprint(corpus, hyperparameters(args), base_hash)
            state = read_state(args.root)
            model = state.get('model')
            if (args.if_changed and state.get('fingerprint') == print_hash
                    and (not model or (args.root / 'models' / model).is_dir())):
                print('Nothing changed since the last run.')
                return 0
            return train(args, corpus, counts, base_hash, print_hash)
    except Busy as error:
        print(f'{error}; this run stops.', file=sys.stderr)
        return 4
    except InputError as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
