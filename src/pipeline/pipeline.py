"""abstract pipeline"""

import abc
import pandas as pd
import typing


_Pipeline = typing.TypeVar(name="_Pipeline", bound="Pipeline")


class Pipeline(abc.ABC):
    """abstract pipeline object"""

    def __init__(self: _Pipeline, id_experiment: int = 0, iteration: int = 0):
        """
        Initializes the pipeline object.

        Args:
            self (_Pipeline): Class instance.
            id_experiment (int, optional): ID of the experiment. Defaults to 0.
            iteration (int, optional): Iteration number. Defaults to 0.
        """
        self.id_experiment = id_experiment
        self.iteration = iteration

    @abc.abstractmethod
    def full_pipeline(
        self: _Pipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        """
        Full pipeline (learning and testing).

        Args:
            self (_Pipeline): Class instance.
            df_train (pd.DataFrame): Training dataframe.
            df_valid (pd.DataFrame): Validation dataframe.
            df_test (pd.DataFrame): Test dataframe.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def learning_pipeline(
        self: _Pipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        """
        Learning pipeline to train a model.

        Args:
            self (_Pipeline): Class instance.
            df_train (pd.DataFrame): Training dataframe.
            df_valid (pd.DataFrame): Validation dataframe.
            df_test (pd.DataFrame): Test dataframe.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def testing_pipeline(
        self: _Pipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        """
        Testing pipeline to evaluate a pre-trained model.

        Args:
            self (_Pipeline): Class instance.
            df_train (pd.DataFrame): Training dataframe.
            df_valid (pd.DataFrame): Validation dataframe.
            df_test (pd.DataFrame): Test dataframe.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError
