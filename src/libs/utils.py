"""Utils functions"""

import os
import torch
from typing import Any
import stat


def on_rm_error(func, path, exc_info):
    """
    Ignore errors when removing files.
    """
    print(f"Error when deleting file {path}: {exc_info[1]}")


def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """

    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def move_to_cpu(obj: Any) -> Any:
    """
    Move an object to CPU.

    Args:
        obj (Any): Object.

    Returns:
        Any: Object moved to CPU.
    """
    if isinstance(obj, torch.nn.Module):
        return obj.to("cpu")
    elif isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {key: move_to_cpu(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(val) for val in obj]
    else:
        return obj
