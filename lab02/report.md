# Lab 2: A Neural Network from Scratch

## 1. Goal

The goal of this lab was to implement a two-hidden-layer neural network from scratch using NumPy. The lab covered forward propagation, softmax, cross-entropy loss, backpropagation, numerical gradient checking, training experiments, and timing measurements.

## 2. Method

The `load_digits` dataset was used. Input pixel values were scaled by dividing by 16, and labels were one-hot encoded. The dataset was split into training and testing sets using a 70/30 stratified split with `random_state=42`.

The network architecture was:

- Input layer: 64 features
- Hidden layer 1: W neurons with ReLU activation
- Hidden layer 2: W neurons with ReLU activation
- Output layer: 10 neurons with softmax activation

Weights were initialized from a normal distribution \(N(0, 0.01)\), and biases were initialized to zero.

Cross-entropy loss and backpropagation were implemented manually. Numerical gradient checking compared analytical gradients with central finite-difference estimates using a step size of \(10^{-5}\). Nine out of ten selected weights passed the relative-error threshold of \(10^{-6}\). The checked weights were selected from parameters with analytical gradient magnitudes greater than \(10^{-6}\).

Training used mini-batches of 32 examples for 50 epochs. The learning rates were 0.001, 0.01, and 0.1, and the hidden widths were 16 and 64.

Forward and backward timing were measured separately using 100 runs after 10 warm-up runs. Median execution times were recorded.

## 3. Results

### Dataset preparation

| Property | Result |
|---|---:|
| Total samples | 1797 |
| Training samples | 1257 |
| Test samples | 540 |
| Input features | 64 |
| Output classes | 10 |

The initial untrained network loss was approximately 2.3026, close to \(\ln(10)\).

### Training results

| Hidden width | Learning rate | Final training loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.3026 | 10.19% |
| 16 | 0.01 | 2.3026 | 10.19% |
| 16 | 0.1 | 1.5646 | 32.59% |
| 64 | 0.001 | 2.3025 | 10.19% |
| 64 | 0.01 | 2.3025 | 10.19% |
| 64 | 0.1 | 0.1586 | 93.89% |

The loss curves are saved in `results/loss_curves.png`, and the detailed results are saved in `results/training_results.csv`.

### Timing results

| Hidden width | Forward median (ms) | Backward median (ms) |
|---:|---:|---:|
| 16 | 0.0327 | 0.0345 |
| 64 | 0.0489 | 0.0613 |

The timing results are saved in `results/timing_results.csv`.

## 4. Conclusions

1. The neural network forward pass, softmax, cross-entropy loss, and manual backpropagation were implemented successfully. Numerical gradient checking achieved the required threshold for 9 out of 10 selected weights.

2. The best result was achieved with hidden width 64 and learning rate 0.1, reaching 93.89% test accuracy. Lower learning rates did not provide sufficient learning within 50 epochs in these experiments.

3. Increasing hidden width from 16 to 64 increased both forward and backward execution times. Backward propagation was slower than forward propagation for both tested widths, consistent with the additional gradient calculations required during training.