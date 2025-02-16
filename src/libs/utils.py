"""Utils functions"""

import torch
from typing import Any


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
