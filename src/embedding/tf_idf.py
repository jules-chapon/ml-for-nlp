"""TD-IDF TfIdfEmbedding class"""

import numpy as np
import typing

from sklearn.feature_extraction.text import TfidfVectorizer

from src.embedding.embedding import Embedding

from src.configs import names

_TfIdfEmbedding = typing.TypeVar(name="_TfIdfEmbedding", bound="TfIdfEmbedding")


class TfIdfEmbedding(Embedding):
    def __init__(self: _TfIdfEmbedding, id_experiment: int = 0):
        super().__init__(id_experiment=id_experiment)
        self.embedding = TfidfVectorizer(
            max_features=self.params[names.MAX_FEATURES], smooth_idf=True
        )

    def fit(self: _TfIdfEmbedding, X: np.ndarray) -> None:
        self.embedding.fit(X)

    def transform(self: _TfIdfEmbedding, X: np.ndarray) -> np.ndarray:
        return self.embedding.transform(X)
