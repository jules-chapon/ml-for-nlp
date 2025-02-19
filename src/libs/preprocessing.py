"""Functions for preprocessing."""

import numpy as np
import pandas as pd
import time

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.configs import constants, names


def load_data_from_hf(
    dataset: int = 1,
    type: str = "full",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load data from Hugging Face dataset.

    Args:
        dataset (int): The dataset to load. Defaults to 1.
        type (str): Which files to load. Defaults to "full".

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: (df_train, df_test).
    """
    start_time = time.time()
    if dataset == 1:
        filename = constants.HF_FULL_DATASET_1_NAME
    else:
        raise ValueError("Invalid dataset")
    if type == "full":
        df_train = load_dataset(filename)["train"].to_pandas()
        df_test = load_dataset(filename)["test"].to_pandas()
    elif type == "train":
        df_train = load_dataset(filename)["train"].to_pandas()
        df_test = None
    elif type == "test":
        df_train = None
        df_test = load_dataset(filename)["test"].to_pandas()
    else:
        raise ValueError("Invalid type")
    print(f"Data loading done in {time.time() - start_time:.2f} seconds")
    return df_train, df_test


def load_data_from_local(
    dataset: int = 1,
    type: str = "full",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load data from local files.

    Args:
        dataset (int): The dataset to load. Defaults to 1.
        type (str): Which files to load. Defaults to "full".

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: (df_train, df_test).
    """
    start_time = time.time()
    if dataset == 1:
        train_path = constants.LOCAL_TRAIN_DATASET_1_PATH
        test_path = constants.LOCAL_TEST_DATASET_1_PATH
    else:
        raise ValueError("Invalid dataset")
    if type == "full":
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    elif type == "train":
        df_train = pd.read_csv(train_path)
        df_test = None
    elif type == "test":
        df_train = None
        df_test = pd.read_csv(test_path)
    else:
        raise ValueError("Invalid type")
    print(f"Data loading done in {time.time() - start_time:.2f} seconds")
    return df_train, df_test


def load_data(
    local: bool = True, dataset: int = 1, type: str = "full"
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load data from either local files or Hugging Face datasets.

    Args:
        local (bool): Whether to load data from local CSV files. Defaults to True.
        dataset (int): The dataset to load. Defaults to 1.
        type (str): Which files to load to load. Defaults to "full".

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: (df_train, df_test).
    """
    if local:
        print("Loading data from local")
        df_train, df_test = load_data_from_local(dataset=dataset, type=type)
    else:
        print("Loading data from Hugging Face")
        df_train, df_test = load_data_from_hf(dataset=dataset, type=type)
    return df_train, df_test


def clean_dataset_1(df: pd.DataFrame) -> pd.DataFrame:
    df[names.TEXT] = df[names.ABSTRACT].str.replace("\n", " ")
    df.drop(names.ABSTRACT, axis=1, inplace=True)
    return df


def split_features_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the DataFrame into features and labels.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple[np.ndarray, np.ndarray]: The features and labels as numpy arrays.
    """
    X = df[names.TEXT].copy().to_numpy()
    y = df[names.LABEL].copy().to_numpy()
    return X, y


def train_valid_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The training and validation DataFrames.
    """
    df_train, df_valid = train_test_split(
        df, test_size=constants.VALID_RATIO, random_state=constants.RANDOM_SEED
    )
    return df_train, df_valid
