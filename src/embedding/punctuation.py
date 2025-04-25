"""PunctuationEmbedding class"""

import numpy as np
import pandas as pd
import typing

from typing import Any

from src.embedding.embedding import Embedding

from src.configs import names

_PunctuationEmbedding = typing.TypeVar(
    name="_PunctuationEmbedding", bound="PunctuationEmbedding"
)


class PunctuationEmbedding(Embedding):
    """PunctuationEmbedding class.

    Attributes:
        id_experiment (int): The experiment ID to use for configuration.
        params (dict): The parameters for the embedding.
        embedding (PunctuationVectorizer): The Punctuation vectorizer object.
    """

    def __init__(self: _PunctuationEmbedding, id_experiment: int = 0):
        """Initialize the Punctuation embedding.

        Args:
            id_experiment (int): The experiment ID to use for configuration.
        """
        super().__init__(id_experiment=id_experiment)

    def fit(self: _PunctuationEmbedding, X: Any) -> None:
        """Raises error.

        Args:
            X (Any): Anything.
        """
        raise NotImplementedError

    def transform(self: _PunctuationEmbedding, X: np.ndarray) -> np.ndarray:
        """Transform the data using the Punctuation vectorizer.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The transformed data.
        """
        X_series = pd.Series(X).astype(str)
        array_nb_characters = X_series.str.len().to_numpy()
        array_ratio_punctuations = (
            X_series.str.count(r"[^\w\s]").to_numpy() / array_nb_characters
        )
        array_ratio_words_per_sentence = (
            X_series.str.count(r"[.!?]").to_numpy()
            / X_series.str.split().str.len().to_numpy()
        )
        array_ratio_odd_characters = (
            X_series.str.count(r"[!;-_]").to_numpy() / array_nb_characters
        )
        embedding = pd.DataFrame(
            {
                "ratio_punctuations": array_ratio_punctuations,
                "ratio_words_per_sentence": array_ratio_words_per_sentence,
                "ratio_odd_characters": array_ratio_odd_characters,
            }
        ).to_numpy()
        return embedding

    def map_dimensions_to_names(self: _PunctuationEmbedding) -> np.ndarray:
        """
        Maps the dimensions of the embedding to their corresponding feature names.

        Returns:
            np.ndarray: An array of feature names corresponding to the dimensions of the embedding.
        """
        return np.array(
            ["ratio_punctuations", "ratio_words_per_sentence", "ratio_odd_characters"]
        )
