"""Functions for preprocessing."""

import numpy as np
import pandas as pd
import time
import os
import shutil

from git import Repo
from sklearn.model_selection import train_test_split

from src.configs import constants, names

from src.libs.utils import on_rm_error, onerror


def download_data() -> None:
    """
    Download data from GitHub and save it in the data folder.
    """
    # Check if destination or temporary folders already exist
    if os.path.exists(constants.DATA_GITHUB_FOLDER):
        print(f"Destination folder {constants.DATA_GITHUB_FOLDER} already exists.")
        return
    if os.path.exists(constants.TEMPORARY_FOLDER):
        print(f"Temporary folder {constants.TEMPORARY_FOLDER} already exists.")
        return

    try:
        # Clone GitHub repo in temporary folder
        Repo.clone_from(constants.REPO_URL, constants.TEMPORARY_FOLDER)
        source_folder = os.path.join(
            constants.TEMPORARY_FOLDER, constants.TARGET_FOLDER
        )

        # Check if target folder exists in GitHub repo
        if not os.path.exists(source_folder):
            print(f"Target folder {source_folder} does not exist in the cloned repo.")
            shutil.rmtree(constants.TEMPORARY_FOLDER, onerror=on_rm_error)
            return

        # Copy the source folder to the destination folder
        shutil.copytree(source_folder, constants.DATA_GITHUB_FOLDER)
        print(f"Folder copied to {constants.DATA_GITHUB_FOLDER}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Delete temporary folder and "file" file if they exist
        if "file" in os.listdir(constants.DATA_GITHUB_FOLDER):
            os.remove(os.path.join(constants.DATA_GITHUB_FOLDER, "file"))
        if os.path.exists(constants.TEMPORARY_FOLDER):
            shutil.rmtree(constants.TEMPORARY_FOLDER, onerror=onerror)
            print(f"Temporary folder {constants.TEMPORARY_FOLDER} removed.")


def load_dataset(source: str = "Human", type: str = "ESSAY") -> pd.DataFrame:
    """
    Load a dataset based on the specified source and type.

    Args:
        source (str): The source of the dataset (e.g., "Human", "GPT", "BARD").
        type (str): The type of the dataset (e.g., "ESSAY", "POETRY", "STORY").

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.

    Raises:
        ValueError: If the source or type is invalid.
    """
    start_time = time.time()
    if source == "Human":
        if type == "ESSAY":
            filename = constants.HUMAN_ESSAY_1
        elif type == "POETRY":
            filename = constants.HUMAN_POETRY
        elif type == "STORY":
            filename = constants.HUMAN_STORY
        else:
            raise ValueError("Invalid type")
    elif source == "GPT":
        if type == "ESSAY":
            filename = constants.GPT_ESSAY
        elif type == "POETRY":
            filename = constants.GPT_POETRY
        elif type == "STORY":
            filename = constants.GPT_STORY
        else:
            raise ValueError("Invalid type")
    elif source == "BARD":
        if type == "ESSAY":
            filename = constants.BARD_ESSAY
        elif type == "POETRY":
            filename = constants.BARD_POETRY
        elif type == "STORY":
            filename = constants.BARD_STORY
        else:
            raise ValueError("Invalid type")
    else:
        raise ValueError("Invalid source")
    df = pd.read_csv(
        os.path.join(constants.DATA_GITHUB_FOLDER, source, filename), index_col=0
    )
    if source == "GPT":
        df.reset_index(drop=False, inplace=True)
    end_time = time.time()
    print(f"Time taken to load dataset: {end_time - start_time} seconds")
    return df


def clean_text_column(col: pd.Series, source: str | None = None) -> pd.Series:
    """
    Clean a text column based on the specified source.

    Args:
        col (pd.Series): The text column to clean.
        source (str | None): The source of the text column (e.g., "BARD").

    Returns:
        pd.Series: The cleaned text column.
    """
    col = pd.Series(
        np.where(
            col.str.startswith("Sure"),
            col.str.split(r"\r\n\r").str[1:].str.join(" "),
            col.str.split(r"\r\n\r").str.join(" "),
        )
    )
    col = col.str.replace("Chapter Text", "")
    col = col.str.replace("\r\r\n", " ")
    col = col.str.replace("\r\n\r", " ")
    col = col.str.replace("\n", " ")
    col = col.str.replace("\r", " ")
    col = col.str.replace(r"\s{2,}", " ", regex=True)
    col = col.str.strip()

    return col


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by removing empty rows, duplicates, and specific text patterns.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df[df[names.TEXT] != ""]
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[~df[names.TEXT].str.contains("ChatGPT")]
    df = df[~df[names.TEXT].str.contains("BARD")]
    return df


def get_poetry_dataset(
    df_gpt_poetry: pd.DataFrame,
    df_bard_poetry: pd.DataFrame,
    df_human_poetry: pd.DataFrame,
    samples_per_source: int | None = None,
) -> pd.DataFrame:
    """
    Get a poetry dataset by cleaning and concatenating datasets from different sources.

    Args:
        df_gpt_poetry (pd.DataFrame): The GPT poetry dataset.
        df_bard_poetry (pd.DataFrame): The BARD poetry dataset.
        df_human_poetry (pd.DataFrame): The human poetry dataset.
        samples_per_source (int | None): The number of samples to take from each source.

    Returns:
        pd.DataFrame: The concatenated and cleaned poetry dataset.
    """
    # Clean GPT
    df_gpt_poetry[names.TEXT] = clean_text_column(
        col=df_gpt_poetry["responses"], source=None
    )
    df_gpt_poetry[names.ABSTRACT] = df_gpt_poetry["prompts"]
    df_gpt_poetry.drop(columns=["prompts", "responses"], inplace=True)
    df_gpt_poetry = clean_dataframe(df_gpt_poetry)

    # Clean BARD
    df_bard_poetry[names.TEXT] = clean_text_column(
        col=df_bard_poetry["BARD"], source="BARD"
    )
    df_bard_poetry[names.ABSTRACT] = df_bard_poetry["prompts"]
    df_bard_poetry.drop(columns=["prompts", "BARD"], inplace=True)
    df_bard_poetry = clean_dataframe(df_bard_poetry)

    # Clean Human
    df_human_poetry[names.TEXT] = clean_text_column(
        col=df_human_poetry["Poem"], source=None
    )
    df_human_poetry[names.ABSTRACT] = df_human_poetry["Title"]
    df_human_poetry.drop(
        columns=["Title", "Poem", "Poet", "Tags"],
        inplace=True,
    )
    df_human_poetry = clean_dataframe(df_human_poetry)

    # Concatenate all dataframes
    df_gpt_poetry[names.LABEL] = constants.GPT_LABEL
    df_bard_poetry[names.LABEL] = constants.BARD_LABEL
    df_human_poetry[names.LABEL] = constants.HUMAN_LABEL
    if samples_per_source is None:
        df_poetry = pd.concat([df_gpt_poetry, df_bard_poetry, df_human_poetry], axis=0)
    else:
        df_poetry = pd.concat(
            [
                df_gpt_poetry.sample(
                    n=samples_per_source,
                    replace=True if samples_per_source > len(df_gpt_poetry) else False,
                    random_state=constants.RANDOM_SEED,
                ),
                df_bard_poetry.sample(
                    n=samples_per_source,
                    replace=True if samples_per_source > len(df_bard_poetry) else False,
                    random_state=constants.RANDOM_SEED,
                ),
                df_human_poetry.sample(
                    n=samples_per_source,
                    replace=True
                    if samples_per_source > len(df_human_poetry)
                    else False,
                    random_state=constants.RANDOM_SEED,
                ),
            ],
            axis=0,
        ).sample(frac=1, random_state=constants.RANDOM_SEED)
    df_poetry.reset_index(drop=True, inplace=True)
    df_poetry[names.TYPE] = names.TYPE_POETRY
    return df_poetry


def get_essay_dataset(
    df_gpt_essay: pd.DataFrame,
    df_bard_essay: pd.DataFrame,
    df_human_essay: pd.DataFrame,
    samples_per_source: int | None = None,
) -> pd.DataFrame:
    """
    Get an essay dataset by cleaning and concatenating datasets from different sources.

    Args:
        df_gpt_essay (pd.DataFrame): The GPT essay dataset.
        df_bard_essay (pd.DataFrame): The BARD essay dataset.
        df_human_essay (pd.DataFrame): The human essay dataset.
        samples_per_source (int | None): The number of samples to take from each source.

    Returns:
        pd.DataFrame: The concatenated and cleaned essay dataset.
    """
    # Clean GPT
    df_gpt_essay[names.TEXT] = clean_text_column(
        col=df_gpt_essay["responses"], source=None
    )
    df_gpt_essay[names.ABSTRACT] = df_gpt_essay["prompts"]
    df_gpt_essay.drop(columns=["prompts", "responses"], inplace=True)
    df_gpt_essay = clean_dataframe(df_gpt_essay)

    # Clean BARD
    df_bard_essay[names.TEXT] = clean_text_column(
        col=df_bard_essay["BARD"], source="BARD"
    )
    df_bard_essay[names.ABSTRACT] = df_bard_essay["prompts"]
    df_bard_essay.drop(columns=["prompts", "BARD"], inplace=True)
    df_bard_essay = clean_dataframe(df_bard_essay)

    # Clean Human
    df_human_essay[names.TEXT] = clean_text_column(
        col=df_human_essay["essays"], source=None
    )
    df_human_essay[names.ABSTRACT] = "No abstract"
    df_human_essay.drop(
        columns=["essays"],
        inplace=True,
    )
    df_human_essay = clean_dataframe(df_human_essay)

    # Concatenate all dataframes
    df_gpt_essay[names.LABEL] = constants.GPT_LABEL
    df_bard_essay[names.LABEL] = constants.BARD_LABEL
    df_human_essay[names.LABEL] = constants.HUMAN_LABEL
    if samples_per_source is None:
        df_essay = pd.concat([df_gpt_essay, df_bard_essay, df_human_essay], axis=0)
    else:
        df_essay = pd.concat(
            [
                df_gpt_essay.sample(
                    n=samples_per_source,
                    replace=True if samples_per_source > len(df_gpt_essay) else False,
                    random_state=constants.RANDOM_SEED,
                ),
                df_bard_essay.sample(
                    n=samples_per_source,
                    replace=True if samples_per_source > len(df_bard_essay) else False,
                    random_state=constants.RANDOM_SEED,
                ),
                df_human_essay.sample(
                    n=samples_per_source,
                    replace=True if samples_per_source > len(df_human_essay) else False,
                    random_state=constants.RANDOM_SEED,
                ),
            ],
            axis=0,
        ).sample(frac=1, random_state=constants.RANDOM_SEED)
    df_essay.reset_index(drop=True, inplace=True)
    df_essay[names.TYPE] = names.TYPE_ESSAY
    return df_essay


def get_story_dataset(
    df_gpt_story: pd.DataFrame,
    df_bard_story: pd.DataFrame,
    df_human_story: pd.DataFrame,
    samples_per_source: int | None = None,
) -> pd.DataFrame:
    """
    Get a story dataset by cleaning and concatenating datasets from different sources.

    Args:
        df_gpt_story (pd.DataFrame): The GPT story dataset.
        df_bard_story (pd.DataFrame): The BARD story dataset.
        df_human_story (pd.DataFrame): The human story dataset.
        samples_per_source (int | None): The number of samples to take from each source.

    Returns:
        pd.DataFrame: The concatenated and cleaned story dataset.
    """
    # Clean GPT
    df_gpt_story[names.TEXT] = clean_text_column(
        col=df_gpt_story["Chapter_text"], source=None
    )
    df_gpt_story[names.ABSTRACT] = df_gpt_story["Title"]
    df_gpt_story.drop(columns=["Title", "Chapter_text", "index"], inplace=True)
    df_gpt_story = clean_dataframe(df_gpt_story)

    # Clean BARD
    df_bard_story[names.TEXT] = clean_text_column(
        col=df_bard_story["BARD"], source=None
    )
    df_bard_story[names.ABSTRACT] = df_bard_story["prompts"]
    df_bard_story.drop(columns=["prompts", "BARD"], inplace=True)
    df_bard_story = clean_dataframe(df_bard_story)

    # Clean Human
    df_human_story[names.TEXT] = clean_text_column(
        col=df_human_story["Chapter_text"], source=None
    )
    df_human_story[names.ABSTRACT] = df_human_story["Summary"]
    df_human_story.drop(
        columns=[
            "Title",
            "Author",
            "Fandoms",
            "Required_tags",
            "Datetime",
            "Tags",
            "ChatGPT_Tag",
            "Summary",
            "Language",
            "Words",
            "Chapters",
            "Comments",
            "Kudos",
            "Bookmarks",
            "Hits",
            "Chapter_text",
        ],
        inplace=True,
    )
    df_human_story = clean_dataframe(df_human_story)

    # Concatenate all dataframes
    df_gpt_story[names.LABEL] = constants.GPT_LABEL
    df_bard_story[names.LABEL] = constants.BARD_LABEL
    df_human_story[names.LABEL] = constants.HUMAN_LABEL
    if samples_per_source is None:
        df_story = pd.concat([df_gpt_story, df_bard_story, df_human_story], axis=0)
    else:
        df_story = pd.concat(
            [
                df_gpt_story.sample(
                    n=samples_per_source,
                    replace=True if samples_per_source > len(df_gpt_story) else False,
                    random_state=constants.RANDOM_SEED,
                ),
                df_bard_story.sample(
                    n=samples_per_source,
                    replace=True if samples_per_source > len(df_bard_story) else False,
                    random_state=constants.RANDOM_SEED,
                ),
                df_human_story.sample(
                    n=samples_per_source,
                    replace=True if samples_per_source > len(df_human_story) else False,
                    random_state=constants.RANDOM_SEED,
                ),
            ],
            axis=0,
        ).sample(frac=1, random_state=constants.RANDOM_SEED)
    df_story.reset_index(drop=True, inplace=True)
    df_story[names.TYPE] = names.TYPE_STORY
    return df_story


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


def group_llms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group the labels in the DataFrame to combine GPT and BARD labels.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with grouped labels.
    """
    df[names.LABEL] = np.where(
        df[names.LABEL] == constants.HUMAN_LABEL,
        constants.HUMAN_LABEL,
        constants.GPT_LABEL,
    )
    return df
