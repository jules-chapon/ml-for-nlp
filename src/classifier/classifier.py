"""Abstract class for scikit-learn classifiers"""

import abc
import numpy as np
import typing

from typing import Any

from src.configs import ml_config


_SKLClassifier = typing.TypeVar(name="_SKLClassifier", bound="SKLClassifier")


class SKLClassifier(abc.ABC):
    def __init__(self: _SKLClassifier, id_experiment: int = 0) -> None:
        super().__init__()
        self.id_experiment = id_experiment
        self.params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        self.classifier = None

    def train(self: _SKLClassifier, X: np.ndarray, y: np.ndarray) -> None:
        return self.classifier.fit(X, y)
