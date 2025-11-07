import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embedding_alignment(X_en, Y_fr, R, output_path="results/embedding_alignment.png", n_points=200):
    """
    Visualize English and French embeddings before and after alignment using PCA.

    Args:
        X_en: English embeddings matrix (m, d)
        Y_fr: French embeddings matrix (m, d)
        R: learned transformation matrix (d, d)
        output_path: where to save the plot
        n_points: how many word pairs to visualize (for clarity)
    """
    os.makedirs("results", exist_ok=True)

    # Limit to first n_points for readability
    X_en = X_en[:n_points]
    Y_fr = Y_fr[:n_points]

    # Project English embeddings into French space
    X_en_aligned = X_en @ R

    # Stack for PCA: before and after, plus French
    all_vectors = np.vstack([X_en, Y_fr, X_en_aligned])

    # Reduce to 2D
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_vectors)

    n = X_en.shape[0]

    en_2d = all_2d[:n]
    fr_2d = all_2d[n:2*n]
    aligned_2d = all_2d[2*n:3*n]

    plt.figure(figsize=(8, 6))

    # Original English (before alignment)
    plt.scatter(en_2d[:, 0], en_2d[:, 1], alpha=0.4, label="English (original)", marker="o")

    # French embeddings
    plt.scatter(fr_2d[:, 0], fr_2d[:, 1], alpha=0.4, label="French", marker="x")

    # English after alignment
    plt.scatter(aligned_2d[:, 0], aligned_2d[:, 1], alpha=0.6, label="English (aligned)", marker="^")

    plt.title("Embedding Space Alignment (PCA projection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
