
import csv
import time
from pathlib import Path

import torch

from models import MLP, CNN, AttentionModel


WARMUP_RUNS = 10
BATCH_CONFIGS = [
    (1, 100),
    (128, 20),
]


def measure_median_latency(model, inputs, runs):
    model.eval()

    # Warm-up before measurement
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(inputs)

        timings = []

        for _ in range(runs):
            start = time.perf_counter()
            model(inputs)
            end = time.perf_counter()

            timings.append((end - start) * 1000)

    timings.sort()
    return timings[len(timings) // 2] if len(timings) % 2 else (
        timings[len(timings) // 2 - 1]
        + timings[len(timings) // 2]
    ) / 2


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    results = []

    for name, model in models.items():
        print(f"\nMeasuring {name}...")

        for batch_size, runs in BATCH_CONFIGS:
            inputs = torch.randn(batch_size, 1, 28, 28)

            latency_ms = measure_median_latency(
                model, inputs, runs
            )

            results.append({
                "model": name,
                "batch_size": batch_size,
                "measured_runs": runs,
                "warmup_runs": WARMUP_RUNS,
                "median_latency_ms": latency_ms,
                "latency_per_image_ms": latency_ms / batch_size,
            })

            print(
                f"Batch={batch_size:3d} | "
                f"Runs={runs:3d} | "
                f"Median latency={latency_ms:.4f} ms | "
                f"Per image={latency_ms / batch_size:.4f} ms"
            )

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "latency_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
