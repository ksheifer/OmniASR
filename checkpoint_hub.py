import torch
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="Steveeeeeeen/omniASR-LLM-1B", filename="omniASR-LLM-1B.pt")

# Check the file size on disk
file_size_bytes = os.path.getsize(path)

# Load the model
ckpt = torch.load(path, map_location="cpu")
def get_model_stats(obj):
    total_params = 0
    total_bytes = 0
    first_tensor = None
    def walk(data):
        nonlocal total_params, total_bytes, first_tensor
        if isinstance(data, dict):
            for v in data.values():
                walk(v)
        elif isinstance(data, list):
            for v in data:
                walk(v)
        elif torch.is_tensor(data):
            if first_tensor is None:
                first_tensor = data
            num_el = data.numel()
            total_params += num_el
            total_bytes += num_el * data.element_size()
    walk(obj)
    return total_params, total_bytes, first_tensor

params, size_bytes, tensor = get_model_stats(ckpt)

# Prepare data for output
dtype = tensor.dtype
bytes_per_param = tensor.element_size()
bits_per_param = bytes_per_param * 8
gb_size = size_bytes / 1e9
gib_size = size_bytes / 1024**3
storage_precision = "FP32" if dtype == torch.float32 else "FP16/BF16"

print(f"{'Data Type':<30} {dtype}")
print(f"{'Bits per parameter':<30} {bytes_per_param} bytes ({bits_per_param} bits)")
print(f"{'Precision (Storage)':<30} {storage_precision}")
print(f"{'Precision (Inference)':<30} FP16 / BF16")
print(f"{'Gigabytes':<30} {gb_size:.2f} GB")
print(f"{'Gibibytes':<30} {gib_size:.2f} GiB")

