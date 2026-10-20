
def main():
    # MLP: Linear(784 -> 256)
    mlp_flops = 2 * 784 * 256

    # CNN: first Conv2d(1 -> 16, kernel 3x3, output 28x28)
    cnn_flops = 2 * 3 * 3 * 1 * 16 * 28 * 28

    # Attention: 16 patches, embedding dimension 64
    seq = 16
    d_model = 64

    # Q, K, V linear projections
    qkv_flops = 2 * seq * d_model * 3 * d_model

    # Attention score computation: Q @ K^T
    attention_score_flops = 2 * seq * seq * d_model

    # Weighted sum of values: Attention @ V
    attention_value_flops = 2 * seq * seq * d_model

    print("Hand-calculated representative FLOPs")
    print("-" * 45)
    print(f"MLP Linear(784 -> 256): {mlp_flops:,}")
    print(f"CNN Conv2d(1 -> 16):    {cnn_flops:,}")
    print(f"Attention QKV:          {qkv_flops:,}")
    print(f"Attention scores:       {attention_score_flops:,}")
    print(f"Attention weighted V:   {attention_value_flops:,}")

    attention_total = (
        qkv_flops
        + attention_score_flops
        + attention_value_flops
    )
    print(f"Attention subtotal:     {attention_total:,}")


if __name__ == "__main__":
    main()
