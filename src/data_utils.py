import pandas as pd
import numpy as np


def get_dict(file_name: str) -> dict:
    """
    Load an English–French dictionary from a space-delimited text file.

    Each row: <english> <french>

    Returns:
        dict: {english_word: french_word}
    """
    df = pd.read_csv(file_name, delimiter=" ", header=None)
    etof = {row[0]: row[1] for _, row in df.iterrows()}
    return etof


def get_matrices(en_fr: dict,
                 french_vecs: dict,
                 english_vecs: dict):
    """
    Build aligned matrices of English and French embeddings
    for word pairs that exist in both embedding sets.

    Args:
        en_fr: {english: french}
        french_vecs: {french_word: embedding_vector}
        english_vecs: {english_word: embedding_vector}

    Returns:
        X: np.ndarray of shape (m, d) English embeddings
        Y: np.ndarray of shape (m, d) French embeddings
    """
    X_l = []
    Y_l = []

    english_set = set(english_vecs.keys())
    french_set = set(french_vecs.keys())

    for en_word, fr_word in en_fr.items():
        if en_word in english_set and fr_word in french_set:
            X_l.append(english_vecs[en_word])
            Y_l.append(french_vecs[fr_word])

    X = np.vstack(X_l)
    Y = np.vstack(Y_l)

    return X, Y
