
import numpy as np
from neural_network import NeuralNetwork


def numerical_gradient(model, X, y, parameter, index, epsilon=1e-5):
    param = getattr(model, parameter)
    original_value = param[index]

    param[index] = original_value + epsilon
    loss_plus = model.cross_entropy_loss(y, model.forward(X))

    param[index] = original_value - epsilon
    loss_minus = model.cross_entropy_loss(y, model.forward(X))

    param[index] = original_value

    return (loss_plus - loss_minus) / (2 * epsilon)


def main():
    rng = np.random.default_rng(42)

    # Small batch for gradient checking
    X = rng.random((8, 64))
    labels = rng.integers(0, 10, size=8)
    y = np.eye(10)[labels]

    model = NeuralNetwork(hidden_size=16, seed=42)

    probabilities = model.forward(X)
    loss = model.cross_entropy_loss(y, probabilities)
    analytical_gradients = model.backward(y)

    # Check 10 randomly selected weights
    parameters = ["W1", "W2", "W3"]
    checks = []


    parameters = ["W1", "W2", "W3"]
    checks = []

    # Select random weights with non-negligible analytical gradients
    candidates = []

    for parameter in parameters:
        gradient = analytical_gradients[parameter]
        for index in np.ndindex(gradient.shape):
            if abs(gradient[index]) > 1e-6:
                candidates.append((parameter, index))

    rng.shuffle(candidates)
    selected = candidates[:10]

    for parameter, index in selected:
        analytical = analytical_gradients[parameter][index]
        numerical = numerical_gradient(
            model, X, y, parameter, index, epsilon=1e-5
        )

        relative_error = abs(analytical - numerical) / max(
            1e-12, abs(analytical) + abs(numerical)
        )

        checks.append(relative_error)

        print(
            f"{parameter}{index}: "
            f"analytical={analytical:.10e}, "
            f"numerical={numerical:.10e}, "
            f"relative error={relative_error:.3e}"
        )

    passed = sum(error < 1e-6 for error in checks)

    print(f"\nInitial loss: {loss:.6f}")
    print(f"Passed: {passed}/10")
    print("Required: at least 9/10 with relative error < 1e-6")


    passed = sum(error < 1e-6 for error in checks)

    print(f"\nInitial loss: {loss:.6f}")
    print(f"Passed: {passed}/10")
    print(f"Required: at least 9/10 with relative error < 1e-6")


if __name__ == "__main__":
    main()
