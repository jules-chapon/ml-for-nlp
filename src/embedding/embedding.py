"""Abstract class for embedding models"""

import abc
import typing

from typing import Any

from src.configs import ml_config

_Embedding = typing.TypeVar(name="_Embedding", bound="Embedding")


class Embedding(abc.ABC):
    """Abstract class for embedding models

    Attributes:
        id_experiment (int): ID of the experiment. Defaults to 0.
        params (dict): Parameters of the experiment. Defaults to ml_config.EXPERIMENTS_CONFIGS[id_experiment].
        embedding (Any): Embedding model. Defaults to None.
    """

    def __init__(self: _Embedding, id_experiment: int = 0):
        """
        Initialize the embedding model.

        Args:
            id_experiment (int, optional): ID of the experiment. Defaults to 0.
        """
        super().__init__()
        self.id_experiment = id_experiment
        self.params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        self.embedding = None

    @abc.abstractmethod
    def fit(self: _Embedding, X: Any) -> None:
        """
        Fit the embedding model.

        Args:
            X (Any): Input data.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self: _Embedding, X: Any) -> Any:
        """
        Transform the input data using the embedding model.

        Args:
            X (Any): Input data.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            Any: Transformed data.
        """
        raise NotImplementedError
