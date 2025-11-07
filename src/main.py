import pickle
import os
import matplotlib.pyplot as plt

from src.data_utils import get_dict, get_matrices
from src.model_utils import align_embeddings, translate_word
from src.evaluate import test_vocabulary
from src.visualize import plot_embedding_alignment


def main():
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)
    
    # Load embeddings
    en_embeddings = pickle.load(open("./data/en_embeddings.p", "rb"))
    fr_embeddings = pickle.load(open("./data/fr_embeddings.p", "rb"))

    # Load dictionaries
    en_fr_train = get_dict("./data/en-fr.train.txt")
    en_fr_test = get_dict("./data/en-fr.test.txt")

    # Build train matrices
    X_train, Y_train = get_matrices(en_fr_train, fr_embeddings, en_embeddings)

    # Train mapping
    R, losses = align_embeddings(X_train, Y_train,
                         train_steps=400,
                         learning_rate=0.8,
                         verbose=True)
    

    # Save loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss for Embedding Alignment")
    plt.tight_layout()
    plt.savefig("results/loss_curve.png")
    plt.close()
    
    
    # Evaluate
    X_val, Y_val = get_matrices(en_fr_test, fr_embeddings, en_embeddings)
    acc = test_vocabulary(X_val, Y_val, R)
    print(f"\nAccuracy on test dictionary: {acc:.3f}\n")
    
    # Visualize alignment on a subset of word pairs
    plot_embedding_alignment(X_train, Y_train, R, output_path="results/embedding_alignment.png")

    # Show a few sample predictions
    print("Sample translations (from test dictionary):")
    sample_items = list(en_fr_test.items())[:10]
    for en_word, true_fr in sample_items:
        pred_fr = translate_word(en_word, en_embeddings, fr_embeddings, R)
        print(f"{en_word:15s} -> predicted: {pred_fr:15s} | true: {true_fr}")
    print()

    # Save a small report
    with open("results/report.txt", "w", encoding="utf-8") as f:
        f.write(f"Final loss: {losses[-1]:.4f}\n")
        f.write(f"Accuracy on test dictionary: {acc:.3f}\n")

    # Demo loop
    while True:
        word = input("Enter an English word (or 'q' to quit): ").strip()
        if word.lower() == "q":
            break
        fr = translate_word(word, en_embeddings, fr_embeddings, R)
        print(f"{word} → {fr}\n")


if __name__ == "__main__":
    main()
