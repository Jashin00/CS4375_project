import torch

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
    print("\n✅ You are good to run 04_bert.py!")
else:
    print("\n❌ No GPU detected — use Google Colab instead")