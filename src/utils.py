from pathlib import Path
import os
import torch
import logging

logger = logging.getLogger('__name__')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_git_root() -> str:
    return Path(os.popen("git rev-parse --show-toplevel").read().strip())


def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str,
):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(f"{get_git_root()}/models")
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    logging.info(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
