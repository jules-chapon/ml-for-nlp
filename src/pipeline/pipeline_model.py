"""Pipeline for Transformer model"""

import numpy as np
import os
import pandas as pd
import pickle as pkl
import time
import torch
import typing

from typing import Any, Tuple

from src.configs import constants, ml_config, names

from src.libs import utils

from src.pipeline.pipeline import Pipeline


_ModelPipeline = typing.TypeVar(name="_ModelPipeline", bound="ModelPipeline")


class ModelPipeline(Pipeline):
    def __init__(
        self: _ModelPipeline, id_experiment: int = 0, iteration: int = 0
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_ModelPipeline): Class instance.
            id_experiment (int, optional): ID of the experiment. Defaults to 0.
            iteration (int, optional): Iteration of the experiment. Defaults to 0.
        """
        super().__init__(id_experiment=id_experiment)
        self.id_experiment
        self.iteration = iteration
        self.params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        self.metrics = {}
        self.training_time = 0.0
        if (self.params[names.DEVICE] == "cuda") and (torch.cuda.is_available()):
            self.params[names.DEVICE] = "cuda"
        else:
            self.params[names.DEVICE] = "cpu"
        self.folder_name = (
            f"{self.params[names.MODEL_TYPE]}_{id_experiment}_{iteration}"
        )
        os.makedirs(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "training"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "test"),
            exist_ok=True,
        )
        print("Pipeline initialized successfully")

    def full_pipeline(
        self: _ModelPipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        """
        Run the full pipeline to train the model and save the results.

        Args:
            self (_ModelPipeline): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Validation set.
            df_test (pd.DataFrame): Test set.
        """
        print("Full pipeline ran successfully")

    def learning_pipeline(
        self: _ModelPipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        raise NotImplementedError

    def testing_pipeline(
        self: _ModelPipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        raise NotImplementedError

    def get_model(
        self: _ModelPipeline,
    ) -> None:
        """
        Get the model associated to the ID of the experiment.

        Args:
            self (_ModelPipeline): Class instance.
        """
        print(f"Model {self.params[names.MODEL_TYPE]} loaded successfully")

    def preprocess_data(
        self: _ModelPipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess raw data before training.

        Args:
            self (_ModelPipeline): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Validation set.
            df_test (pd.DataFrame): Test set.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (Train set, validation set, test set).
        """
        print("Dataloaders loaded successfully")
        return df_train, df_valid, df_test

    def save(self: _ModelPipeline) -> None:
        """
        Save the instance.

        Args:
            self (_ModelPipeline): Class instance.
        """
        path = os.path.join(
            constants.OUTPUT_FOLDER, self.folder_name, "training", "pipeline.pkl"
        )
        self.model = self.model.to("cpu")
        self.metrics = utils.move_to_cpu(self.metrics)
        self.training_time = utils.move_to_cpu(self.training_time)
        self.src_vocab = utils.move_to_cpu(self.src_vocab)
        self.src_vocab_reversed = utils.move_to_cpu(self.src_vocab_reversed)
        self.tgt_vocab = utils.move_to_cpu(self.tgt_vocab)
        self.tgt_vocab_reversed = utils.move_to_cpu(self.tgt_vocab_reversed)
        self_to_cpu = utils.move_to_cpu(self)
        with open(path, "wb") as file:
            pkl.dump(self_to_cpu, file)
        print("Model saved successfully")

    def save_losses(self, train_loss: list[float], valid_loss: list[float]) -> None:
        """
        Save train and validation losses.

        Args:
            train_loss (list[float]): Train losses.
            valid_loss (list[float]): Validation losses.
        """
        train_loss = utils.move_to_cpu(train_loss)
        valid_loss = utils.move_to_cpu(valid_loss)
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "training", "train_loss.npy"
            ),
            train_loss,
        )
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "training", "valid_loss.npy"
            ),
            valid_loss,
        )
        print("Losses saved successfully")

    def load(self: _ModelPipeline) -> None:
        """
        Load a pre-trained model.

        Args:
            self (_ModelPipeline): Class instance.
        """
        print(
            f"Model {self.params[names.MODEL_TYPE]} number {self.iteration-1} of experiment {self.id_experiment} loaded successfully"
        )
