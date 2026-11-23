"""Render the timestamped meeting tracks into one locally playable audio file."""
from pathlib import Path
from fractions import Fraction
import wave


RATE = 16000


def track_blocks(folder, chunks, total):
    """One second of mono PCM at a time, preserving gaps and trimming overlapping timestamps."""
    import numpy as np

    rows = iter(chunks)
    row = next(rows, None)
    samples = None
    for cursor in range(0, total, RATE):
        block = np.zeros(min(RATE, total - cursor), dtype=np.int32)
        while row is not None:
            start = int(row['startMs']) * 16
            if start >= cursor + len(block):
                break
            if samples is None:
                try:
                    with wave.open(str(folder / row['file']), 'rb') as wav:
                        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
                            raise wave.Error('Unsupported meeting audio format')
                        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
                except (EOFError, wave.Error):
                    row = next(rows, None)
                    continue
            left, right = max(start, cursor), min(start + len(samples), cursor + len(block))
            if right > left:
                block[left-cursor:right-cursor] = samples[left-start:right-start]
            if start + len(samples) > cursor + len(block):
                break
            row, samples = next(rows, None), None
        yield block


def export_recording(folder, chunks, seconds):
    import av
    import numpy as np

    folder = Path(folder)
    total = max(round(seconds * RATE), max((r['startMs'] + r['durationMs']) * 16 for r in chunks))
    rows = sorted(chunks, key=lambda r: r['startMs'])
    tracks = [track_blocks(folder, [r for r in rows if r['source'] == source], total)
              for source in sorted({r['source'] for r in rows})]
    output = folder / 'meeting.m4a'
    temporary = folder / 'meeting.m4a.tmp'
    try:
        with av.open(str(temporary), mode='w', format='mp4') as container:
            stream = container.add_stream('aac', rate=RATE)
            stream.layout = 'mono'
            stream.bit_rate = 64000
            cursor = 0
            for blocks in zip(*tracks):
                pcm = np.clip(sum(blocks), -32768, 32767).astype(np.int16)
                frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format='s16', layout='mono')
                frame.sample_rate = RATE
                frame.pts = cursor
                frame.time_base = Fraction(1, RATE)
                cursor += frame.samples
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
