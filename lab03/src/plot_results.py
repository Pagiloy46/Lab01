
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    lab_dir = Path(__file__).resolve().parents[1]
    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    training = read_csv(results_dir / "training_results.csv")
    latency = read_csv(results_dir / "latency_results.csv")

    latency_batch1 = {
        row["model"]: float(row["median_latency_ms"])
        for row in latency
        if int(row["batch_size"]) == 1
    }

    model_names = [row["model"] for row in training]
    parameters = [int(row["parameters"]) for row in training]
    accuracy = [float(row["test_accuracy"]) * 100 for row in training]
    latency_values = [latency_batch1[name] for name in model_names]

    # Accuracy vs parameter count
    plt.figure(figsize=(8, 6))
    for name, params, acc in zip(model_names, parameters, accuracy):
        plt.scatter(params, acc, s=80, label=name)
        plt.annotate(name, (params, acc), xytext=(5, 5),
                     textcoords="offset points")

    plt.xlabel("Trainable parameters")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs Parameter Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_vs_parameters.png", dpi=150)
    plt.close()

    # Accuracy vs single-image latency
    plt.figure(figsize=(8, 6))
    for name, latency_ms, acc in zip(
        model_names, latency_values, accuracy
    ):
        plt.scatter(latency_ms, acc, s=80, label=name)
        plt.annotate(name, (latency_ms, acc), xytext=(5, 5),
                     textcoords="offset points")

    plt.xlabel("Median inference latency for batch 1 (ms)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs Inference Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_vs_latency.png", dpi=150)
    plt.close()

    print("Plots saved:")
    print(results_dir / "accuracy_vs_parameters.png")
    print(results_dir / "accuracy_vs_latency.png")


if __name__ == "__main__":
    main()
