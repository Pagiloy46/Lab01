
from pathlib import Path

import torch
from torchinfo import summary
from thop import profile

from models import MLP, CNN, AttentionModel


def main():
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    # One Fashion-MNIST image
    dummy_input = torch.randn(1, 1, 28, 28)

    results = []

    for name, model in models.items():
        model.eval()

        print("\n" + "=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)

        # Parameter count and layer-by-layer information
        model_summary = summary(
            model,
            input_size=(1, 1, 28, 28),
            verbose=0,
        )

        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print(model_summary)
        print(f"Trainable parameters: {trainable_params:,}")

        # THOP profiling
        with torch.no_grad():
            macs, thop_params = profile(
                model,
                inputs=(dummy_input,),
                verbose=False,
            )

        print(f"THOP MACs: {macs:,.0f}")
        print(f"THOP parameters: {thop_params:,.0f}")
        print(
            "Approximate FLOPs from THOP MACs "
            f"(2 FLOPs per MAC): {2 * macs:,.0f}"
        )

        results.append({
            "model": name,
            "parameters": trainable_params,
            "thop_macs": macs,
            "approx_flops_from_macs": 2 * macs,
        })

    # Save results to CSV
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "model_costs.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        import csv

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "parameters",
                "thop_macs",
                "approx_flops_from_macs",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
