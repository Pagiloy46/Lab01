import os
import time
import threading
import statistics

import joblib
import psutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


RANDOM_STATE = 42
RESULT_DIR = os.path.join("result")


# ---------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------

def monitor_memory(stop_event, memory_samples):
    process = psutil.Process(os.getpid())

    while not stop_event.is_set():
        memory_samples.append(process.memory_info().rss)
        time.sleep(0.001)

    memory_samples.append(process.memory_info().rss)


def measure_with_memory(function):
    memory_samples = []
    stop_event = threading.Event()

    monitor = threading.Thread(
        target=monitor_memory,
        args=(stop_event, memory_samples),
        daemon=True
    )

    monitor.start()

    start_time = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start_time

    stop_event.set()
    monitor.join()

    peak_memory = max(memory_samples)

    return result, elapsed, peak_memory


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

# One fixed sample for single-sample inference
X_single = X_test[0:1]


# ---------------------------------------------------------
# Model definitions
# ---------------------------------------------------------

def create_logistic():
    return LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )


def create_random_forest():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )


# ---------------------------------------------------------
# Training benchmark
# ---------------------------------------------------------

def benchmark_training(model_factory, name):
    print(f"\n=== {name} training ===")

    # Warm-up
    print("Warm-up...")
    warmup_model = model_factory()
    warmup_model.fit(X_train, y_train)

    training_times = []
    peak_memories = []
    last_model = None

    # 5 measured runs
    for run in range(5):
        model = model_factory()

        model, elapsed, peak_memory = measure_with_memory(
            lambda: model.fit(X_train, y_train)
        )

        training_times.append(elapsed)
        peak_memories.append(peak_memory)

        last_model = model

        print(
            f"Run {run + 1}: "
            f"{elapsed:.6f} s, "
            f"peak memory {peak_memory / (1024 ** 2):.2f} MB"
        )

    median_training_time = statistics.median(training_times)
    peak_training_memory = max(peak_memories)

    print(f"Median training time: {median_training_time:.6f} s")
    print(
        f"Peak training memory: "
        f"{peak_training_memory / (1024 ** 2):.2f} MB"
    )

    return last_model, median_training_time, peak_training_memory


# ---------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------

def benchmark_inference(model, name):
    print(f"\n=== {name} inference ===")

    inference_times = []
    peak_memories = []

    # Warm-up inference
    model.predict(X_single)

    # 100 single-sample inference runs
    for _ in range(100):
        _, elapsed, peak_memory = measure_with_memory(
            lambda: model.predict(X_single)
        )

        inference_times.append(elapsed)
        peak_memories.append(peak_memory)

    median_inference_time = statistics.median(inference_times)
    peak_inference_memory = max(peak_memories)

    print(
        f"Median single-sample inference: "
        f"{median_inference_time * 1000:.6f} ms"
    )

    print(
        f"Peak inference memory: "
        f"{peak_inference_memory / (1024 ** 2):.2f} MB"
    )

    return median_inference_time, peak_inference_memory


# ---------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------

os.makedirs(RESULT_DIR, exist_ok=True)

results = []

models = [
    ("LogisticRegression", create_logistic),
    ("RandomForest", create_random_forest),
]

for name, model_factory in models:

    model, training_time, training_memory = benchmark_training(
        model_factory,
        name
    )

    # Accuracy check
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Inference benchmark
    inference_time, inference_memory = benchmark_inference(
        model,
        name
    )

    # Save model
    model_path = os.path.join(
        RESULT_DIR,
        f"{name}.joblib"
    )

    joblib.dump(model, model_path)

    # Model size
    model_size_bytes = os.path.getsize(model_path)
    model_size_kb = model_size_bytes / 1024

    results.append({
        "model": name,
        "accuracy": accuracy,
        "training_time_median_s": training_time,
        "inference_latency_median_ms": inference_time * 1000,
        "model_size_bytes": model_size_bytes,
        "model_size_kb": model_size_kb,
        "peak_training_memory_mb": training_memory / (1024 ** 2),
        "peak_inference_memory_mb": inference_memory / (1024 ** 2),
    })


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_path = os.path.join(
    RESULT_DIR,
    "system_measurements.csv"
)

with open(results_path, "w", encoding="utf-8") as file:
    file.write(
        "model,accuracy,training_time_median_s,"
        "inference_latency_median_ms,model_size_bytes,"
        "model_size_kb,peak_training_memory_mb,"
        "peak_inference_memory_mb\n"
    )

    for result in results:
        file.write(
            f"{result['model']},"
            f"{result['accuracy']:.4f},"
            f"{result['training_time_median_s']:.6f},"
            f"{result['inference_latency_median_ms']:.6f},"
            f"{result['model_size_bytes']},"
            f"{result['model_size_kb']:.2f},"
            f"{result['peak_training_memory_mb']:.2f},"
            f"{result['peak_inference_memory_mb']:.2f}\n"
        )


# ---------------------------------------------------------
# Print final table
# ---------------------------------------------------------

print("\n================ FINAL RESULTS ================")

for result in results:
    print(f"\nModel: {result['model']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(
        f"Median training time: "
        f"{result['training_time_median_s']:.6f} s"
    )
    print(
        f"Median inference latency: "
        f"{result['inference_latency_median_ms']:.6f} ms"
    )
    print(
        f"Model size: "
        f"{result['model_size_bytes']} bytes "
        f"({result['model_size_kb']:.2f} KB)"
    )
    print(
        f"Peak training memory: "
        f"{result['peak_training_memory_mb']:.2f} MB"
    )
    print(
        f"Peak inference memory: "
        f"{result['peak_inference_memory_mb']:.2f} MB"
    )

print(f"\nResults saved to: {results_path}")