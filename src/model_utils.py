import numpy as np


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors (or matrix row vs vector).
    """
    dot = np.dot(A, B)
    norm_b = np.linalg.norm(B)

    if A.ndim == 1:
        norm_a = np.linalg.norm(A)
        return dot / (norm_a * norm_b)
    else:
        norm_a = np.linalg.norm(A, axis=1)
        eps = 1e-9
        return dot / (norm_a * norm_b + eps)


def nearest_neighbor(v: np.ndarray,
                     candidates: np.ndarray,
                     k: int = 1,
                     similarity_fn=cosine_similarity):
    """
    Find indices of k most similar rows in `candidates` to vector `v`.
    """
    sims = [similarity_fn(row, v) for row in candidates]
    sorted_ids = np.argsort(sims)[::-1]
    return sorted_ids[:k]


def compute_loss(X: np.ndarray, Y: np.ndarray, R: np.ndarray) -> float:
    """
    Mean squared error loss ||XR - Y||^2 / m
    """
    m = X.shape[0]
    diff = X @ R - Y
    return np.sum(diff ** 2) / m


def compute_gradient(X: np.ndarray, Y: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Gradient of loss wrt R: 2/m * X^T (XR - Y)
    """
    m = X.shape[0]
    return (2 / m) * X.T @ (X @ R - Y)


def align_embeddings(X: np.ndarray,
                     Y: np.ndarray,
                     train_steps: int = 400,
                     learning_rate: float = 0.8,
                     verbose: bool = True) -> np.ndarray:
    """
    Learn transformation R that maps English to French space
    by minimizing ||XR - Y||^2 via gradient descent.
    """
    np.random.seed(129)
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        if verbose and i % 25 == 0:
            print(f"Iteration {i} - loss: {compute_loss(X, Y, R):.4f}")
        grad = compute_gradient(X, Y, R)
        R -= learning_rate * grad

    if verbose:
        print(f"Final loss: {compute_loss(X, Y, R):.4f}")

    return R


def translate_word(english_word: str,
                   en_embeddings: dict,
                   fr_embeddings: dict,
                   R: np.ndarray) -> str:
    """
    Translate an English word to French using learned mapping.
    """
    if english_word not in en_embeddings:
        return f"[OOV] '{english_word}' not found in English embeddings."

    en_vec = en_embeddings[english_word]
    projected = en_vec @ R

    min_dist = float("inf")
    best_word = None

    for fr_word, fr_vec in fr_embeddings.items():
        dist = np.linalg.norm(projected - fr_vec)
        if dist < min_dist:
            min_dist = dist
            best_word = fr_word

    return best_word
