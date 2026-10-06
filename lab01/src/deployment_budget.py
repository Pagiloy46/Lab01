# Measurements from system_measurements.csv

models = {
    "LogisticRegression": {
        "latency_ms": 0.302000,
        "size_kb": 1.03,
        "memory_mb": 104.50,
    },
    "RandomForest": {
        "latency_ms": 2.610750,
        "size_kb": 284.07,
        "memory_mb": 106.10,
    },
}

budgets = {
    "Cloud": {
        "max_latency_ms": 100,
        "max_size_kb": 500 * 1024,
        "min_memory_mb": 1024,
    },
    "Edge": {
        "max_latency_ms": 50,
        "max_size_kb": 50 * 1024,
        "min_memory_mb": 256,
    },
    "Mobile": {
        "max_latency_ms": 20,
        "max_size_kb": 10 * 1024,
        "min_memory_mb": 64,
    },
    "TinyML": {
        "max_latency_ms": 10,
        "max_size_kb": 100,
        "max_memory_kb": 256,
    },
}


def fits(model, platform):
    latency_ok = model["latency_ms"] <= platform["max_latency_ms"]
    size_ok = model["size_kb"] <= platform["max_size_kb"]

    if "min_memory_mb" in platform:
        memory_ok = model["memory_mb"] <= platform["min_memory_mb"]
    else:
        memory_ok = model["memory_mb"] * 1024 <= platform["max_memory_kb"]

    return latency_ok and size_ok and memory_ok


print("Deployment suitability")
print("=" * 80)

for model_name, model in models.items():
    print(f"\n{model_name}")
    print("-" * 80)

    for platform_name, platform in budgets.items():
        result = "YES" if fits(model, platform) else "NO"
        print(f"{platform_name:10} -> {result}")