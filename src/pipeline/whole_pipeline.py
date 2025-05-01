import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

###############
### PUNCTUATION
###############


def punctuation_transform(X: np.ndarray) -> np.ndarray:
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


###############
### TFIDF
###############


def punctuation_map_dimensions_to_names() -> np.ndarray:
    return np.array(
        ["ratio_punctuations", "ratio_words_per_sentence", "ratio_odd_characters"]
    )


def fit_tfidf_embedding(X_train_texts, max_features=1000):
    """
    Fit a TF-IDF vectorizer on training texts.

    Args:
        X_train_texts (list or np.ndarray): Training texts.
        max_features (int): Maximum number of features for TF-IDF.

    Returns:
        vectorizer: Fitted TfidfVectorizer instance.
        X_train_tfidf: Transformed training data.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features, smooth_idf=True, stop_words="english"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_texts)
    return vectorizer, X_train_tfidf


def transform_with_tfidf(vectorizer, X_texts):
    """
    Transform texts using an already fitted TF-IDF vectorizer.

    Args:
        vectorizer: A fitted TfidfVectorizer.
        X_texts (list or np.ndarray): Texts to transform.

    Returns:
        X_tfidf: Transformed TF-IDF matrix.
    """
    return vectorizer.transform(X_texts)


###############
### FINE TUNING
###############


def train_fine_tune_RF(X_train, y_train):
    ### Random search

    param_distributions = {
        "n_estimators": randint(50, 300),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
        "bootstrap": [True, False],
    }

    base_rf = RandomForestClassifier(n_jobs=-1, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    search.fit(X_train, y_train)

    best_rf = search.best_estimator_

    print("Best hyperparameters:", search.best_params_)
    return best_rf


###############
### EVALUATION
###############


def eval_model_binary(
    classifier,
    X_train,
    y_train,
    X_test,
    y_test,
    X_test_poetry,
    y_test_poetry,
    X_test_essay,
    y_test_essay,
    X_test_story,
    y_test_story,
):
    ### CLASSIFIER
    classifier.fit(X_train, y_train)

    # Evaluation
    metrics = evaluate(
        classifier, X_test, y_test, display_labels=["Human", "AI"], title="Test set"
    )
    metrics_poetry = evaluate(
        classifier,
        X_test_poetry,
        y_test_poetry,
        display_labels=["Human", "AI"],
        title="Poetry",
    )
    metrics_essay = evaluate(
        classifier,
        X_test_essay,
        y_test_essay,
        display_labels=["Human", "AI"],
        title="Essay",
    )
    metrics_story = evaluate(
        classifier,
        X_test_story,
        y_test_story,
        display_labels=["Human", "AI"],
        title="Story",
    )

    results = {
        "test": metrics,
        "poetry": metrics_poetry,
        "essay": metrics_essay,
        "story": metrics_story,
    }
    df_metrics = pd.DataFrame(results).T
    print(df_metrics.round(3))
    return classifier


def eval_model(
    classifier,
    X_train,
    y_train,
    X_test,
    y_test,
    X_test_poetry,
    y_test_poetry,
    X_test_essay,
    y_test_essay,
    X_test_story,
    y_test_story,
):
    ### CLASSIFIER
    classifier.fit(X_train, y_train)

    # Evaluation
    metrics = evaluate(
        classifier,
        X_test,
        y_test,
        display_labels=["Human", "GPT", "Bard"],
        title="Test set",
    )
    metrics_poetry = evaluate(
        classifier,
        X_test_poetry,
        y_test_poetry,
        display_labels=["Human", "GPT", "Bard"],
        title="Poetry",
    )
    metrics_essay = evaluate(
        classifier,
        X_test_essay,
        y_test_essay,
        display_labels=["Human", "GPT", "Bard"],
        title="Essay",
    )
    metrics_story = evaluate(
        classifier,
        X_test_story,
        y_test_story,
        display_labels=["Human", "GPT", "Bard"],
        title="Story",
    )

    results = {
        "test": metrics,
        "poetry": metrics_poetry,
        "essay": metrics_essay,
        "story": metrics_story,
    }
    df_metrics = pd.DataFrame(results).T
    print(df_metrics.round(3))
    return classifier


def evaluate(model, X, y, display_labels=None, title=None):
    """
    Evaluate a model, display the confusion matrix, and return evaluation scores.

    Args:
        model: Trained sklearn-like model.
        X (np.ndarray): Input features.
        y (np.ndarray): True labels.
        display_labels (list, optional): Class names for display.

    Returns:
        dict: Dictionary containing confusion matrix, precision, recall, f1 score.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    if display_labels is None:
        classes = np.unique(np.concatenate([y, y_pred]))
        display_labels = [str(label) for label in classes]

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {title}")
    plt.tight_layout()
    plt.show()

    return {
        "confusion_matrix": cm,
        "precision": precision_score(y, y_pred, average="macro"),
        "recall": recall_score(y, y_pred, average="macro"),
        "f1": f1_score(y, y_pred, average="macro"),
    }


###############
### GET OUT OF FOLD
###############

def get_oof_predictions(X, y, model_func, n_splits=5):
    """
    Generate out-of-fold predictions for a given model, to be used as features in a meta-classifier.

    Args:
        X (np.ndarray or pd.DataFrame): Input features.
        y (np.ndarray or pd.Series): Target labels.
        model_func (callable): A function that takes (X_train, y_train) and returns a fitted model.
        n_splits (int): Number of folds in cross-validation.

    Returns:
        np.ndarray: OOF predictions aligned with the input order (shape: [n_samples, 1]).
    """
    oof_preds = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train the model on the training split
        model = model_func(X_train, y_train)
        
        # Predict probabilities on the validation (out-of-fold) split
        # Use the probability of class 1 (AI-generated)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Return as column vector for stacking
    return oof_preds.reshape(-1, 1)

def get_oof_predictions_tfidf(X, y, model_func, n_splits=5):
    """
    Generate out-of-fold predictions for a given model, to be used as features in a meta-classifier.

    Args:
        X (np.ndarray or sparse matrix): Input features.
        y (np.ndarray or pd.Series): Target labels.
        model_func (callable): A function that takes (X_train, y_train) and returns a fitted model.
        n_splits (int): Number of folds in cross-validation.

    Returns:
        np.ndarray: OOF predictions aligned with the input order (shape: [n_samples, 1]).
    """
    oof_preds = np.zeros(X.shape[0]) 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_func(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    return oof_preds.reshape(-1, 1)