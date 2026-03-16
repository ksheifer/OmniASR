import os
import torchaudio

audio_file = "/Users/karinasheifer/Documents/UCSB/ASR/cluster_testing/ko7o_ya7vi.wav"
output_dir = "/Users/karinasheifer/Documents/UCSB/ASR/cluster_testing/ko7o_ya7vi"
chunk_length = 35.0
overlap = 1         # overlap on both ends

os.makedirs(output_dir, exist_ok=True)

# waveform - audiodata tensor; sample rate - discretization frequency (samples per second)
waveform, sample_rate = torchaudio.load(audio_file)
audio_duration = waveform.shape[1] / sample_rate
print(f"Audio length: {audio_duration:.2f} секунд")

# Splitting logic
start = 0.0

# Calculate the window with overlap
while start < audio_duration:
    start_bi = max(start - overlap, 0.0) # ensures we don't start before 0.0
    end_bi = min(start + chunk_length + overlap, audio_duration) # ensures we don't exceed the file length

    # Convert time (seconds) to frame indices (integers)
    start_frame = int(start_bi * sample_rate)
    end_frame = int(end_bi * sample_rate)

    # Slice the waveform tensor to get the specific chunk
    chunk_waveform = waveform[:, start_frame:end_frame]
    chunk_filename = os.path.join(
        output_dir,
        # Format the filename with padded zeros
        f"ko7o_ya7vi_{int(start_bi):04d}-{int(end_bi):04d}.wav"
    )

    torchaudio.save(chunk_filename, chunk_waveform, sample_rate)
    print(f"Saved in: {chunk_filename}")

    start += chunk_length

print(f"Done. Saved in '{output_dir}'")
