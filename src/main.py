import pickle
from src.data_utils import get_dict, get_matrices
from src.model_utils import align_embeddings, translate_word
from src.evaluate import test_vocabulary



def main():
    # Load embeddings
    en_embeddings = pickle.load(open("./data/en_embeddings.p", "rb"))
    fr_embeddings = pickle.load(open("./data/fr_embeddings.p", "rb"))

    # Load dictionaries
    en_fr_train = get_dict("./data/en-fr.train.txt")
    en_fr_test = get_dict("./data/en-fr.test.txt")

    # Build train matrices
    X_train, Y_train = get_matrices(en_fr_train, fr_embeddings, en_embeddings)

    # Train mapping
    R = align_embeddings(X_train, Y_train,
                         train_steps=400,
                         learning_rate=0.8,
                         verbose=True)

    # Evaluate
    X_val, Y_val = get_matrices(en_fr_test, fr_embeddings, en_embeddings)
    acc = test_vocabulary(X_val, Y_val, R)
    print(f"\nAccuracy on test dictionary: {acc:.3f}\n")

    # Demo loop
    while True:
        word = input("Enter an English word (or 'q' to quit): ").strip()
        if word.lower() == "q":
            break
        fr = translate_word(word, en_embeddings, fr_embeddings, R)
        print(f"{word} → {fr}\n")


if __name__ == "__main__":
    main()
