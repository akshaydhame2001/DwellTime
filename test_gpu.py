import torch

def check_gpu_availability():
  """Checks if a CUDA-enabled GPU is available for PyTorch."""
  if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"PyTorch is using CUDA.")
    print(f"Number of GPUs available: {gpu_count}")
    for i in range(gpu_count):
      print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    return True
  else:
    print("PyTorch is using CPU.")
    return False

if __name__ == "__main__":
  is_gpu_available = check_gpu_availability()
  print(f"Is GPU available for PyTorch? {is_gpu_available}")