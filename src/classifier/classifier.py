"""Abstract class for scikit-learn classifiers"""

import abc
import lightgbm as lgb
import numpy as np
import shap
import typing

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB

from src.configs import constants, ml_config, names
from src.libs import evaluation


_Classifier = typing.TypeVar(name="_Classifier", bound="Classifier")


class Classifier(abc.ABC):
    """
    Abstract base class for basic classifiers.

    Attributes:
        id_experiment (int): The experiment ID to use for configuration.
        params (dict): The parameters for the classifier.
        classifier (object): The classifier object.
    """

    def __init__(self: _Classifier, id_experiment: int = 0) -> None:
        """
        Initialize the classifier.

        Args:
            id_experiment (int): The experiment ID to use for configuration.
        """
        super().__init__()
        self.id_experiment = id_experiment
        self.params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        self.classifier = None

    def get_cv_score(
        self: _Classifier, X_train: np.ndarray, y_train: np.ndarray
    ) -> float:
        """
        Get the cross-validated F1 score of the classifier.

        Args:
            X_train (np.ndarray): The training input samples.
            y_train (np.ndarray): The target values.

        Returns:
            float: The mean cross-validated F1 score.
        """
        ### TODO : add kfold to make sure same randomness
        return cross_val_score(
            self.classifier, X_train, y_train, cv=5, n_jobs=-1, scoring="f1"
        ).mean()

    def train(self: _Classifier, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classifier.

        Args:
            X_train (np.ndarray): The training input samples.
            y_train (np.ndarray): The target values.
        """
        self.classifier.fit(X_train, y_train)

    def predict(self: _Classifier, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return self.classifier.predict(X)

    def evaluate(self: _Classifier, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """
        Evaluate the classifier and return performance metrics.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The true labels.

        Returns:
            dict[str, float]: A dictionary of performance metrics.
        """
        y_pred = self.predict(X)
        return evaluation.get_metrics(y_true=y, y_pred=y_pred)

    def get_shap_values(self: _Classifier, X: np.ndarray) -> np.ndarray:
        explainer = shap.Explainer(self.classifier)
        shap_values = explainer(X)
        return shap_values.values


_LightGBMClassifier = typing.TypeVar(
    name="_LightGBMClassifier", bound="LightGBMClassifier"
)


class LightGBMClassifier(Classifier):
    """
    LightGBM classifier.

    Attributes:
        id_experiment (int): The experiment ID to use for configuration.
        params (dict): The parameters for the classifier.
        classifier (lgb.LGBMClassifier): The LightGBM classifier object.
    """

    def __init__(self: _LightGBMClassifier, id_experiment: int = 0) -> None:
        """
        Initialize the LightGBM classifier.

        Args:
            id_experiment (int): The experiment ID to use for configuration.
        """
        super().__init__(id_experiment)
        self.classifier = lgb.LGBMClassifier(**self.params[names.CLASSIFIER_PARAMS])

    def get_feature_importance(self: _LightGBMClassifier) -> np.ndarray:
        return self.classifier.feature_importances_


_RFClassifier = typing.TypeVar(name="_RFClassifier", bound="RFClassifier")


class RFClassifier(Classifier):
    """
    Random Forest classifier.

    Attributes:
        id_experiment (int): The experiment ID to use for configuration.
        params (dict): The parameters for the classifier.
        classifier (RandomForestClassifier): The Random Forest classifier object.
    """

    def __init__(self: _RFClassifier, id_experiment: int = 0) -> None:
        """
        Initialize the Random Forest classifier.

        Args:
            id_experiment (int): The experiment ID to use for configuration.
        """
        super().__init__(id_experiment)
        self.classifier = RandomForestClassifier(**self.params[names.CLASSIFIER_PARAMS])


_NBClassifier = typing.TypeVar(name="_NBClassifier", bound="NBClassifier")


class NBClassifier(Classifier):
    """
    Naive Bayes classifier.

    Attributes:
        id_experiment (int): The experiment ID to use for configuration.
        params (dict): The parameters for the classifier.
        classifier (BernoulliNB): The Naive Bayes classifier object.
    """

    def __init__(self: _NBClassifier, id_experiment: int = 0) -> None:
        """
        Initialize the Naive Bayes classifier.

        Args:
            id_experiment (int): The experiment ID to use for configuration.
        """
        super().__init__(id_experiment)
        self.classifier = BernoulliNB(**self.params[names.CLASSIFIER_PARAMS])
