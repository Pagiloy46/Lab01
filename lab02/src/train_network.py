
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Allow importing neural_network.py from this folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neural_network import NeuralNetwork


SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATES = [0.001, 0.01, 0.1]
HIDDEN_WIDTHS = [16, 64]


def prepare_data():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train, learning_rate, seed):
    rng = np.random.default_rng(seed)
    n_samples = X_train.shape[0]
    epoch_losses = []

    for epoch in range(EPOCHS):
        # Shuffle training examples each epoch
        indices = rng.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        batch_losses = []

        for start in range(0, n_samples, BATCH_SIZE):
            X_batch = X_shuffled[start:start + BATCH_SIZE]
            y_batch = y_shuffled[start:start + BATCH_SIZE]

            probabilities = model.forward(X_batch)
            loss = model.cross_entropy_loss(y_batch, probabilities)
            gradients = model.backward(y_batch)

            # Gradient descent update
            for name, gradient in gradients.items():
                parameter = getattr(model, name)
                parameter -= learning_rate * gradient

            batch_losses.append(loss)

        # Mean loss across batches for this epoch
        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)

        print(
            f"Width={model.W1.shape[1]:2d}, "
            f"LR={learning_rate:g}, "
            f"Epoch={epoch + 1:02d}/{EPOCHS}, "
            f"Loss={epoch_loss:.4f}"
        )

    return epoch_losses


def accuracy(model, X, y):
    probabilities = model.forward(X)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = np.argmax(y, axis=1)
    return float(np.mean(predictions == true_labels))


def main():
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "lab02" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_data()

    experiment_results = []
    curves = {}

    for width in HIDDEN_WIDTHS:
        for learning_rate in LEARNING_RATES:
            print(
                f"\nStarting experiment: "
                f"hidden_width={width}, learning_rate={learning_rate}"
            )

            model = NeuralNetwork(
                input_size=64,
                hidden_size=width,
                output_size=10,
                seed=SEED,
            )

            losses = train_model(
                model,
                X_train,
                y_train,
                learning_rate,
                seed=SEED,
            )

            test_accuracy = accuracy(model, X_test, y_test)
            label = f"W={width}, lr={learning_rate:g}"
            curves[label] = losses

            experiment_results.append({
                "hidden_width": width,
                "learning_rate": learning_rate,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "final_train_loss": losses[-1],
                "test_accuracy": test_accuracy,
            })

            print(
                f"Finished: {label}, "
                f"final loss={losses[-1]:.4f}, "
                f"test accuracy={test_accuracy:.4f}"
            )

    # Save experiment results
    csv_path = results_dir / "training_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "hidden_width",
                "learning_rate",
                "epochs",
                "batch_size",
                "final_train_loss",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(experiment_results)


    # Plot loss curves with distinct styles
    plt.figure(figsize=(12, 7))

    styles = [
        ("-", "o"),
        ("--", "s"),
        ("-.", "^"),
        (":", "D"),
        ("--", "v"),
        ("-", "P"),
    ]

    for (label, losses), (linestyle, marker) in zip(
        curves.items(), styles
    ):
        plt.plot(
            range(1, EPOCHS + 1),
            losses,
            label=label,
            linestyle=linestyle,
            marker=marker,
            markevery=5,
            markersize=4,
            linewidth=1.5,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = results_dir / "loss_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()



if __name__ == "__main__":
    main()
