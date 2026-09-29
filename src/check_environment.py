import sys
import torch

print("Python version:")
print(sys.version)

print("\nPyTorch version:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    print("\nCUDA version:")
    print(torch.version.cuda)

    print("\nGPU count:")
    print(torch.cuda.device_count())

else:
    print("\nNo GPU detected.")
    print("Training will use CPU.")
