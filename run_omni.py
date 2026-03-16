from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = ASRInferencePipeline(
    model_card="omniASR_LLM_7B",
    device=device
)

chunk_dir = "/home/sheifer/asr/balyksyt_vad"

audio_files = [
    os.path.join(chunk_dir, f)
    for f in os.listdir(chunk_dir)
    if f.endswith(".wav")
]
audio_files.sort()
lang = ["sah_Cyrl"] * len(audio_files)

results = pipeline.transcribe(
    audio_files,
    lang=lang,
    batch_size=1
)

for wav_path, transcription in zip(audio_files, results):
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    txt_path = os.path.join(chunk_dir, f"{base_name}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcription.strip() + "\n")
