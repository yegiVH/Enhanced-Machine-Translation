import numpy as np
from .model_utils import nearest_neighbor


def test_vocabulary(X, Y, R) -> float:
    """
    Evaluate mapping accuracy:
    For each English embedding in X, map with R and
    check if nearest neighbor in Y is at same index.
    """
    pred = X @ R
    num_correct = 0

    for i in range(pred.shape[0]):
        idx = nearest_neighbor(pred[i], Y, k=1)[0]
        if idx == i:
            num_correct += 1

    return num_correct / pred.shape[0]
