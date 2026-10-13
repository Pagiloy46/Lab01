
import csv
import time
import sys
from pathlib import Path

import numpy as np
from neural_network import NeuralNetwork


def median_time(function, runs=100):
    # Warm-up runs
    for _ in range(10):
        function()

    measurements = []

    for _ in range(runs):
        start = time.perf_counter()
        function()
        end = time.perf_counter()
        measurements.append((end - start) * 1000)

    return float(np.median(measurements))


def main():
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "lab02" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    X_batch = rng.random((32, 64))
    labels = rng.integers(0, 10, size=32)
    y_batch = np.eye(10)[labels]

    rows = []

    for width in (16, 64):
        model = NeuralNetwork(hidden_size=width, seed=42)

        # Measure forward pass only
        forward_ms = median_time(
            lambda: model.forward(X_batch)
        )

        # Prepare cache before timing backward
        model.forward(X_batch)

        # Measure backward pass only
        backward_ms = median_time(
            lambda: model.backward(y_batch)
        )

        rows.append({
            "hidden_width": width,
            "batch_size": 32,
            "runs": 100,
            "forward_median_ms": forward_ms,
            "backward_median_ms": backward_ms,
        })

        print(f"\nHidden width: {width}")
        print(f"Forward median:  {forward_ms:.6f} ms")
        print(f"Backward median: {backward_ms:.6f} ms")

    csv_path = results_dir / "timing_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTiming results saved to: {csv_path}")


if __name__ == "__main__":
    main()
