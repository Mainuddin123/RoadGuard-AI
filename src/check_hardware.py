
import platform
import psutil
import torch

print("ROADGUARD AI - HARDWARE CHECK")

print("\nProcessor:", platform.processor())
print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPU cores:", psutil.cpu_count(logical=True))

memory = psutil.virtual_memory()

print("Total RAM:", round(memory.total / (1024 ** 3), 2), "GB")
print("Available RAM:", round(memory.available / (1024 ** 3), 2), "GB")

print("\nPyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
