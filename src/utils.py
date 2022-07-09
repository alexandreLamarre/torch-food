from pathlib import Path
import os
import torch

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_git_root() -> str:
    return Path(os.popen("git rev-parse --show-toplevel").read().strip())
