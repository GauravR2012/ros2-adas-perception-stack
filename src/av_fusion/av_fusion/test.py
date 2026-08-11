import torch
import numpy as np

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", np.__version__)
print("CPU test:", torch.tensor([1.0, 2.0, 3.0]).sum().item())

import spconv
print("spconv:", spconv.__version__)