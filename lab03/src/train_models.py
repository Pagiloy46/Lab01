
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from models import MLP, CNN, AttentionModel
from prepare_data import prepare_data


SEED = 42
EPOCHS = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 128


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_one_model(name, model, train_loader, val_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total = 0

        start_time = time.perf_counter()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / total
        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )
        elapsed = time.perf_counter() - start_time

        print(
            f"{name} | Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    parameters = sum(p.numel() for p in model.parameters())

    print(
        f"\n{name} final test accuracy: {test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)\n"
    )

    return {
        "model": name,
        "parameters": parameters,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Batch size:", BATCH_SIZE)

    train_loader, val_loader, test_loader = prepare_data(
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    models = [
        ("MLP", MLP()),
        ("CNN", CNN()),
        ("Attention", AttentionModel()),
    ]

    results = []

    for name, model in models:
        print(f"\n{'=' * 50}\nTraining {name}\n{'=' * 50}")

        result = train_one_model(
            name,
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
        )
        results.append(result)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "training_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nFinal summary:")
    for result in results:
        print(
            f"{result['model']}: "
            f"parameters={result['parameters']:,}, "
            f"test accuracy={result['test_accuracy']:.4f}"
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
