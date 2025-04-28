"""TD-IDF TfIdfEmbedding class"""

import numpy as np
import typing

from sklearn.feature_extraction.text import TfidfVectorizer

from src.embedding.embedding import Embedding

from src.configs import names

_TfIdfEmbedding = typing.TypeVar(name="_TfIdfEmbedding", bound="TfIdfEmbedding")


class TfIdfEmbedding(Embedding):
    """TD-IDF TfIdfEmbedding class.

    Attributes:
        id_experiment (int): The experiment ID to use for configuration.
        params (dict): The parameters for the embedding.
        embedding (TfidfVectorizer): The TF-IDF vectorizer object.
    """

    def __init__(self: _TfIdfEmbedding, id_experiment: int = 0):
        """Initialize the TF-IDF embedding.

        Args:
            id_experiment (int): The experiment ID to use for configuration.
        """
        super().__init__(id_experiment=id_experiment)
        self.embedding = TfidfVectorizer(
            max_features=self.params[names.EMBEDDING_PARAMS][names.MAX_FEATURES],
            smooth_idf=True,
            stop_words="english",
        )

    def fit(self: _TfIdfEmbedding, X: np.ndarray) -> None:
        """Fit the TF-IDF vectorizer to the training data.

        Args:
            X (np.ndarray): The training data.
        """
        self.embedding.fit(X)

    def transform(self: _TfIdfEmbedding, X: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted TF-IDF vectorizer.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The transformed data.
        """
        return self.embedding.transform(X)

    def map_dimensions_to_names(self: _TfIdfEmbedding) -> np.ndarray:
        """
        Maps the dimensions of the embedding to their corresponding feature names.

        Returns:
            np.ndarray: An array of feature names corresponding to the dimensions of the embedding.
        """
        return self.embedding.get_feature_names_out()
