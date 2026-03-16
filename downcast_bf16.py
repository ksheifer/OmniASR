import torch
import os
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = ASRInferencePipeline(
    model_card="omniASR_LLM_1B",
    device=device
)

# Convert weights to bfloat16
pipeline.model = pipeline.model.to(dtype=torch.bfloat16)

# Define variables for the prints
current_dtype = next(pipeline.model.parameters()).dtype

bits_per_param = 16 
bytes_per_param = bits_per_param // 8

# Save the weights
save_path = "/home/sheifer/asr/omniASR_1B_bf16.pt" 
torch.save(pipeline.model.state_dict(), save_path)

# Verify the result and calculate sizes
if os.path.exists(save_path):
    file_bytes = os.path.getsize(save_path)

    gb_size = file_bytes / (1000**3)   # Gigabytes (decimal)
    gib_size = file_bytes / (1024**3) # Gibibytes (binary)

    print("Model saved")
    print(f"{'Bytes':<30} {file_bytes:,} B")
    print(f"{'Gigabytes':<30} {gb_size:.2f} GB")
    print(f"{'Gibibytes':<30} {gib_size:.2f} GiB")

else:
    print("Error not saved")

