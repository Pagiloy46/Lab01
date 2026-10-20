# Lab 3: Architectures and Their Costs

## 1. Goal

The goal of this laboratory was to implement and compare three neural network architectures in PyTorch: a Multilayer Perceptron (MLP), a Convolutional Neural Network (CNN), and an attention-based Transformer model.

The comparison focuses on test accuracy, trainable parameters, computational cost (FLOPs), inference latency, and architectural access patterns.

## 2. Dataset and Method

The Fashion-MNIST dataset was loaded using `torchvision.datasets.FashionMNIST`.

Images were converted to tensors using `ToTensor()` and normalized using `Normalize((0.2860,), (0.3530,))`.

The official training dataset was split into 55,000 training samples and 5,000 validation samples. The official test dataset contained 10,000 samples. The split used a fixed random seed of 42. The batch size was 128.

All models were trained for five epochs using the Adam optimizer with a learning rate of 0.001.

### 2.1. MLP Architecture

The MLP consisted of the following layers:

- Flatten: 784 input features
- Linear(784, 256)
- ReLU
- Linear(256, 128)
- ReLU
- Linear(128, 10)

### 2.2. CNN Architecture

The CNN consisted of:

- Conv2d(1, 16, kernel_size=3, padding=1)
- ReLU
- MaxPool2d(2)
- Conv2d(16, 32, kernel_size=3, padding=1)
- ReLU
- MaxPool2d(2)
- Flatten
- Linear(32 × 7 × 7, 10)

### 2.3. Attention Architecture

Each 28 × 28 image was divided into 16 patches of 7 × 7 pixels. Each patch was projected from 49 input values to an embedding dimension of 64.

The model used two Transformer encoder layers with `d_model=64`, `nhead=4`, and `dim_feedforward=128`. Mean pooling was applied across the patch sequence, followed by a Linear(64, 10) classifier.

## 3. Parameter Counts

The parameter counts were measured from the implemented PyTorch models.

| Model | Trainable parameters |
|---|---:|
| MLP | 235,146 |
| CNN | 20,490 |
| Attention | 104,266 |

The suggested target was 100,000–150,000 parameters. The Attention model falls within this range, while the MLP exceeds it and the CNN has fewer parameters. The architectures specified in the assignment were retained for comparison.

## 4. Computational Cost and FLOPs

### 4.1. Hand Calculations

For a linear layer, the approximate number of floating-point operations is:

\[
\mathrm{FLOPs} \approx 2C_{\mathrm{in}}C_{\mathrm{out}}
\]

For a convolutional layer:

\[
\mathrm{FLOPs} \approx
2K^2C_{\mathrm{in}}C_{\mathrm{out}}H_{\mathrm{out}}W_{\mathrm{out}}
\]

For the attention mechanism, the QKV projections require approximately:

\[
6SD^2
\]

The attention score calculation and weighted value sum each require approximately:

\[
2S^2D
\]

where \(S\) is the sequence length and \(D\) is the embedding dimension.

Using these formulas, the representative hand-calculated costs were:

| Operation | Approximate FLOPs |
|---|---:|
| MLP Linear(784 → 256) | 401,408 |
| CNN Conv2d(1 → 16, kernel 3 × 3, output 28 × 28) | 225,792 |
| Attention QKV projections | 393,216 |
| Attention score calculation | 32,768 |
| Attention weighted value sum | 32,768 |
| Attention subtotal | 458,752 |

These calculations represent selected operations rather than the complete FLOPs of each model.

### 4.2. Profiler Comparison

The `torchinfo` tool was used to inspect layer shapes and model parameters. The `thop.profile` tool was used to estimate computational cost.

The profiler output was saved in `results/model_costs.csv`.

THOP reports MACs for supported operations. A common approximation is to count one multiply-add as two FLOPs. However, profiler results may omit or count differently some operations, particularly Transformer operations such as attention scores, softmax, and other tensor operations. Consequently, THOP estimates may differ from manual calculations.

The parameter count from the model itself or `torchinfo` is used as the authoritative parameter count. THOP's parameter count for the Attention model is incomplete because not all Transformer parameters are captured by its hooks.

## 5. Training Results

All models were trained for five epochs using Adam with a learning rate of 0.001.

| Model | Trainable parameters | Test loss | Test accuracy |
|---|---:|---:|---:|
| MLP | 235,146 | 0.3368 | 87.74% |
| CNN | 20,490 | 0.2917 | 89.68% |
| Attention | 104,266 | 0.4674 | 83.12% |

CNN achieved the highest test accuracy of 89.68%, followed by MLP at 87.74% and Attention at 83.12%.

The training results were saved in `results/training_results.csv`.

## 6. Inference Latency

Inference latency was measured on CPU. Ten warm-up runs were performed before measurement. The median was calculated from 100 runs for batch size 1 and 20 runs for batch size 128.

| Model | Batch 1 median (ms) | Batch 128 median (ms) | Batch 128 latency per image (ms) |
|---|---:|---:|---:|
| MLP | 0.3759 | 7.1385 | 0.0558 |
| CNN | 2.2804 | 238.7327 | 1.8651 |
| Attention | 4.0934 | 115.2452 | 0.9004 |

MLP had the lowest single-image inference latency. For batch size 128, MLP also had the lowest total latency in this experiment, while CNN had the highest total latency.

The latency measurements were saved in `results/latency_results.csv`. These measurements describe the tested CPU environment and may differ on other hardware.

## 7. Trade-off Plots

Two plots were generated:

- `results/accuracy_vs_parameters.png`
- `results/accuracy_vs_latency.png`

The accuracy-versus-parameters plot compares model quality with model size in terms of trainable parameters. The accuracy-versus-latency plot compares test accuracy with median single-image inference latency.

CNN achieved the highest test accuracy with the fewest trainable parameters. MLP achieved the lowest single-image latency. These results show that the most suitable architecture depends on whether accuracy, parameter count, or inference speed is the primary requirement.

## 8. Architectural Trade-offs and Access Patterns

### MLP

MLP flattens each image into a one-dimensional vector. Its fully connected layers are straightforward to implement and provide low single-image inference latency in this experiment. However, flattening does not explicitly preserve the two-dimensional spatial structure of the image, and the model has the largest parameter count among the three architectures.

### CNN

CNN applies local convolutional filters and shares their weights across image locations. These access patterns exploit spatial locality and preserve local image structure. Pooling reduces the spatial dimensions of feature maps. In this experiment, CNN achieved the highest test accuracy with only 20,490 parameters. Its batch-128 latency was nevertheless higher than that of the other models on the tested CPU.

### Attention

The attention model processes image patches as a sequence and uses self-attention to model interactions between patches. This supports interactions between different image regions, but attention score computation scales quadratically with sequence length. The Transformer also requires projection and feed-forward computations. In this experiment, Attention had the lowest test accuracy and the highest single-image latency.

### Cost-Effectiveness

CNN was the most cost-effective model in terms of test accuracy per parameter. MLP was the best choice for low-latency single-image inference. Attention did not outperform the other models in test accuracy or single-image latency under the tested configuration.

## 9. Conclusions

1. Three architectures, MLP, CNN, and Attention, were implemented in PyTorch and evaluated on Fashion-MNIST using the same data split, batch size, optimizer, learning rate, and number of epochs.

2. CNN achieved the highest test accuracy of 89.68% with 20,490 parameters, making it the strongest accuracy-to-parameter trade-off in this experiment. MLP achieved 87.74% accuracy, while Attention achieved 83.12%.

3. MLP had the lowest single-image inference latency at approximately 0.376 ms. The results demonstrate that architecture selection requires considering accuracy, parameter count, computational cost, latency, and data access patterns together.