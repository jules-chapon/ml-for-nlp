"""Abstract class for embedding models"""

import abc
import typing

from typing import Any

from src.configs import ml_config

_Embedding = typing.TypeVar(name="_Embedding", bound="Embedding")


class Embedding(abc.ABC):
    def __init__(self: _Embedding, id_experiment: int = 0):
        super().__init__()
        self.id_experiment = id_experiment
        self.params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        self.embedding = None

    @abc.abstractmethod
    def fit(self: _Embedding, X: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self: _Embedding, X: Any) -> Any:
        raise NotImplementedError
