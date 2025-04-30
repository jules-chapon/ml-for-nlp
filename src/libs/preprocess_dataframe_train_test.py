# src/libs/data_loader.py

import pandas as pd
from src.libs import preprocessing


def load_all_datasets():
    ### LOAD DATA

    df_poetry_gpt = preprocessing.load_dataset(source="GPT", type="POETRY")
    df_poetry_bard = preprocessing.load_dataset(source="BARD", type="POETRY")
    df_poetry_human = preprocessing.load_dataset(source="Human", type="POETRY")

    df_essay_gpt = preprocessing.load_dataset(source="GPT", type="ESSAY")
    df_essay_bard = preprocessing.load_dataset(source="BARD", type="ESSAY")
    df_essay_human = preprocessing.load_dataset(source="Human", type="ESSAY")

    df_story_gpt = preprocessing.load_dataset(source="GPT", type="STORY")
    df_story_bard = preprocessing.load_dataset(source="BARD", type="STORY")
    df_story_human = preprocessing.load_dataset(source="Human", type="STORY")

    ### GROUPED LLM
    df_poetry = preprocessing.get_poetry_dataset(
        df_poetry_gpt,
        df_poetry_bard,
        df_poetry_human,
        samples_per_source=min(
            len(df_poetry_gpt),
            len(df_poetry_bard),
            len(df_poetry_human),
        ),
    )

    df_essay = preprocessing.get_essay_dataset(
        df_essay_gpt,
        df_essay_bard,
        df_essay_human,
        samples_per_source=min(
            len(df_essay_gpt), len(df_essay_bard), len(df_essay_human)
        ),
    )

    df_story = preprocessing.get_story_dataset(
        df_story_gpt,
        df_story_bard,
        df_story_human,
        samples_per_source=min(
            len(df_story_gpt), len(df_story_bard), len(df_story_human)
        ),
    )

    ### GET TRAIN SPLIT

    df_train_poetry, df_test_poetry = preprocessing.train_valid_split(df_poetry)
    df_train_essay, df_test_essay = preprocessing.train_valid_split(df_essay)
    df_train_story, df_test_story = preprocessing.train_valid_split(df_story)

    ### CONCATENATE ALL DATASETS

    df_train = pd.concat(
        [df_train_poetry, df_train_essay, df_train_story], axis=0
    ).sample(frac=1)
    df_test = pd.concat([df_test_poetry, df_test_essay, df_test_story], axis=0).sample(
        frac=1
    )

    ### SPLIT FEATURES AND LABELS

    X_train, y_train = preprocessing.split_features_and_labels(df_train)
    X_test, y_test = preprocessing.split_features_and_labels(df_test)

    X_train_poetry, y_train_poetry = preprocessing.split_features_and_labels(
        df_train_poetry
    )
    X_train_essay, y_train_essay = preprocessing.split_features_and_labels(
        df_train_essay
    )
    X_train_story, y_train_story = preprocessing.split_features_and_labels(
        df_train_story
    )

    X_test_poetry, y_test_poetry = preprocessing.split_features_and_labels(
        df_test_poetry
    )
    X_test_essay, y_test_essay = preprocessing.split_features_and_labels(df_test_essay)
    X_test_story, y_test_story = preprocessing.split_features_and_labels(df_test_story)

    return {
        ### X, y globaux
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        ### dataframes globaux
        "df_train": df_train,
        "df_test": df_test,
        ### par type : poésie, essai, histoire (train)
        "df_train_poetry": df_train_poetry,
        "df_train_essay": df_train_essay,
        "df_train_story": df_train_story,
        "X_train_poetry": X_train_poetry,
        "y_train_poetry": y_train_poetry,
        "X_train_essay": X_train_essay,
        "y_train_essay": y_train_essay,
        "X_train_story": X_train_story,
        "y_train_story": y_train_story,
        ### par type : poésie, essai, histoire (test)
        "df_test_poetry": df_test_poetry,
        "df_test_essay": df_test_essay,
        "df_test_story": df_test_story,
        "X_test_poetry": X_test_poetry,
        "y_test_poetry": y_test_poetry,
        "X_test_essay": X_test_essay,
        "y_test_essay": y_test_essay,
        "X_test_story": X_test_story,
        "y_test_story": y_test_story,
    }
