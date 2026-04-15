import torch
print("PyTorch 版本:", torch.__version__)        # 预期输出: 1.12.1+cu116 或 1.12.1
print("GPU 是否可用:", torch.cuda.is_available()) # 预期输出: True