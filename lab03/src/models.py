
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 28x28 image -> 16 patches of 7x7 pixels
        self.patch_projection = nn.Linear(49, 64)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=2,
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.shape[0]

        # [batch, 1, 28, 28] -> [batch, 1, 4, 7, 4, 7]
        x = x.reshape(batch_size, 1, 4, 7, 4, 7)

        # Rearrange into 16 patches of 7x7 pixels
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(batch_size, 16, 49)

        # Project each patch to 64 dimensions
        x = self.patch_projection(x)

        # Transformer encoder
        x = self.encoder(x)

        # Mean pool across the 16 patches
        x = x.mean(dim=1)

        # Output logits for 10 classes
        return self.classifier(x)


if __name__ == "__main__":
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    dummy_input = torch.randn(2, 1, 28, 28)

    for name, model in models.items():
        output = model(dummy_input)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )

        print(f"{name}:")
        print("  Output shape:", output.shape)
        print("  Parameters:", f"{parameter_count:,}")
        print()
