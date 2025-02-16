"""Visualization functions"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from ipywidgets import fixed
from typing import List, Literal, Union
from wordcloud import WordCloud

from src.configs import constants


def plot_losses(train_losses: np.ndarray, valid_losses: np.ndarray) -> None:
    """
    Plot the evolution of train and valid losses.

    Args:
        train_losses (np.ndarray): Train losses.
        valid_losses (np.ndarray): Validation losses.
    """
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Validation")
    plt.title("Training and Validation Losses")
    plt.grid(visible=True, which="major", axis="y")
    plt.xlabel("Epoch")
    plt.xticks([x for x in range(0, train_losses.shape[0] - 1) if x % 2 == 0])
    plt.ylabel("Loss")
    plt.ylim((4, 10))
    plt.legend(loc="upper right")
    plt.show()


def plot_histogram_sentence_length(
    sentence_lengths: np.ndarray, type: Literal["words", "characters"] = "words"
) -> None:
    """
    Plot the histogram of sentence lengths in the dataset.

    Args:
        sentence_lengths (np.ndarray): Length of each sentence (whether number of words or characters).
        type (str, optional): Whether considering word or character level. Defaults to "words".
    """
    plt.style.use("science")
    plt.figure(figsize=(10, 6))
    plt.hist(
        sentence_lengths,
        range=(0, 1000) if type == "words" else (0, 10000),
        bins=20,
        weights=np.ones(len(sentence_lengths)) / len(sentence_lengths),
    )
    plt.title(f"Sentence Lengths (in number of {type})")
    plt.xlabel(f"Sentence Length (in number of {type})")
    plt.ylabel("Frequency")
    plt.grid(axis="both", which="major", alpha=0.5)
    plt.show()


def plot_wordcloud(index: int, texts: np.ndarray, labels: np.ndarray) -> None:
    try:
        stopwords = [x.strip() for x in open(constants.STOPWORDS_PATH).readlines()]
    except FileNotFoundError:
        stopwords = None
    text = texts[index]
    label = labels[index]
    wc = WordCloud(background_color="white", repeat=True, stopwords=stopwords)
    wc.generate(text)
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.title("AI-generated text" if label == 1 else "Human-written text")
    plt.imshow(wc, interpolation="bilinear")
    plt.show()


def plot_wordcloud_slider(texts: np.ndarray, labels: np.ndarray) -> None:
    id = widgets.IntSlider(min=0, max=labels.shape[0] - 1, step=1, description="ID")
    widgets.interact(
        plot_wordcloud,
        index=id,
        texts=fixed(texts),
        labels=fixed(labels),
    )
