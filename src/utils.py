from pathlib import Path
import os


def get_git_root() -> str:
    return Path(os.popen("git rev-parse --show-toplevel").read().strip())
