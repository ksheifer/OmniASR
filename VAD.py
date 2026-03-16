import os
import math
import torchaudio

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

audio_file = "/Users/karinasheifer/Documents/UCSB/ASR/cluster_testing/revenge.wav"
output_dir = "/Users/karinasheifer/Documents/UCSB/ASR/cluster_testing/revenge_vad"

MAX_CHUNK_LEN = 40.0          # hard cap
MAX_SILENCE_GAP = 1.5         # if pause is this, try to merge
SHORT_CHUNK_THRESHOLD = 5.0   # try to absorb short chunks into previous one
TARGET_SR = 16000             # for VAD

# VAD tuning
VAD_THRESHOLD = 0.5
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 400
SPEECH_PAD_MS = 150

# Helpers
def sec(x):
    return float(x)

def duration(seg):
    return seg[1] - seg[0]

def print_chunks_info(chunks, title):
    print(f"\n{title}")
    print("-" * len(title))
    for i, (s, e) in enumerate(chunks, 1):
        print(f"{i:03d}: {s:8.3f} -> {e:8.3f}   dur={e-s:8.3f}")
    print(f"Total chunks: {len(chunks)}")


# Load audio and VAD
os.makedirs(output_dir, exist_ok=True)
model = load_silero_vad()
wav = read_audio(audio_file, sampling_rate=TARGET_SR)
audio_duration = len(wav) / TARGET_SR

# Run VAD
speech_timestamps = get_speech_timestamps(
    wav,
    model,
    sampling_rate=TARGET_SR,
    threshold=VAD_THRESHOLD,
    min_speech_duration_ms=MIN_SPEECH_MS,
    min_silence_duration_ms=MIN_SILENCE_MS,
    speech_pad_ms=SPEECH_PAD_MS,
)

if not speech_timestamps:
    print("No speech detected.")
    raise SystemExit

speech_segments = [
    (seg["start"] / TARGET_SR, seg["end"] / TARGET_SR)
    for seg in speech_timestamps
]

print_chunks_info(speech_segments, "Raw speech segments from VAD")

# Pack segments greedily
packed_chunks = []

current_start, current_end = speech_segments[0]

for seg_start, seg_end in speech_segments[1:]:
    silence_gap = seg_start - current_end
    proposed_len = seg_end - current_start

    # Merge if pause is not too large
    # or total chunk length stays <= max
    if silence_gap <= MAX_SILENCE_GAP and proposed_len <= MAX_CHUNK_LEN:
        current_end = seg_end
    else:
        packed_chunks.append((current_start, current_end))
        current_start, current_end = seg_start, seg_end

packed_chunks.append((current_start, current_end))

print_chunks_info(packed_chunks, "Greedy packed chunks")

# Absorb very short chunks into previous one
smoothed_chunks = []

for chunk in packed_chunks:
    start, end = chunk
    dur = end - start

    if not smoothed_chunks:
        smoothed_chunks.append(chunk)
        continue

    prev_start, prev_end = smoothed_chunks[-1]

    # If current chunk is short, try to attach it to previous
    if dur < SHORT_CHUNK_THRESHOLD and (end - prev_start) <= MAX_CHUNK_LEN:
        smoothed_chunks[-1] = (prev_start, end)
    else:
        smoothed_chunks.append(chunk)

print_chunks_info(smoothed_chunks, "After absorbing short chunks into previous")

# If anything is still > 40s, split it
final_chunks = []

for start, end in smoothed_chunks:
    dur = end - start

    if dur <= MAX_CHUNK_LEN:
        final_chunks.append((start, end))
    else:
        # split evenly into pieces <= MAX_CHUNK_LEN
        num_parts = math.ceil(dur / MAX_CHUNK_LEN)
        part_len = dur / num_parts

        sub_start = start
        while sub_start < end:
            sub_end = min(sub_start + part_len, end)
            final_chunks.append((sub_start, sub_end))
            sub_start = sub_end

print_chunks_info(final_chunks, "Final chunks")

# Load original audio for saving chunks
waveform, sample_rate = torchaudio.load(audio_file)

base_name = os.path.splitext(os.path.basename(audio_file))[0]

durations = []

for i, (start_sec, end_sec) in enumerate(final_chunks, 1):
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)

    chunk_waveform = waveform[:, start_frame:end_frame]

    chunk_filename = os.path.join(
        output_dir,
        f"{base_name}_{i:03d}.wav"
    )

    torchaudio.save(chunk_filename, chunk_waveform, sample_rate)
    durations.append(end_sec - start_sec)
    print(f"Saved: {chunk_filename}")

print(f"\nSaved {len(final_chunks)} chunks in '{output_dir}'")
