
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size=64, hidden_size=16, output_size=10, seed=42):
        rng = np.random.default_rng(seed)

        # Layer 1: input -> hidden 1
        self.W1 = rng.normal(0, 0.01, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2: hidden 1 -> hidden 2
        self.W2 = rng.normal(0, 0.01, (hidden_size, hidden_size))
        self.b2 = np.zeros((1, hidden_size))

        # Layer 3: hidden 2 -> output
        self.W3 = rng.normal(0, 0.01, (hidden_size, output_size))
        self.b3 = np.zeros((1, output_size))

        self.cache = {}

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        # Numerical stability
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Hidden layer 1
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        # Hidden layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        # Output layer
        z3 = a2 @ self.W3 + self.b3
        probabilities = self.softmax(z3)

        # Save intermediate values for backpropagation
        self.cache = {
            "X": X,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "z3": z3,
            "probabilities": probabilities,
        }

        return probabilities

    def cross_entropy_loss(self, y_true, probabilities):
        eps = 1e-12
        probabilities = np.clip(probabilities, eps, 1.0)

        return -np.mean(
            np.sum(y_true * np.log(probabilities), axis=1)
        )

    def backward(self, y_true):
        X = self.cache["X"]
        a1 = self.cache["a1"]
        a2 = self.cache["a2"]
        z1 = self.cache["z1"]
        z2 = self.cache["z2"]
        probabilities = self.cache["probabilities"]

        batch_size = X.shape[0]

        # Output layer: softmax + cross-entropy
        dz3 = (probabilities - y_true) / batch_size
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Hidden layer 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    # Prepare dataset
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    # Test forward pass, loss, and backpropagation
    model = NeuralNetwork(hidden_size=16)

    probabilities = model.forward(X_train[:32])
    loss = model.cross_entropy_loss(y_train[:32], probabilities)
    gradients = model.backward(y_train[:32])

    print("Initial loss:", loss)
    print("Expected loss near:", np.log(10))
    print("Output shape:", probabilities.shape)

    print("\nGradient shapes:")
    for name, gradient in gradients.items():
        print(f"{name}: {gradient.shape}")

    print("\nAll gradients contain finite values:",
          all(np.isfinite(g).all() for g in gradients.values()))
