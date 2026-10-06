# Lab 1: Environment and First System Measurements

## 1. Goal

The goal of this lab was to create a reproducible Python environment,
train two baseline machine learning models, measure their system costs,
and evaluate their suitability for different deployment targets.

The models used were Logistic Regression and Random Forest on the
Breast Cancer Wisconsin (Diagnostic) dataset.

## 2. Method

The laboratory environment was created using Python 3.11.8 with pinned
package versions.

The Breast Cancer Wisconsin (Diagnostic) dataset was loaded using
`load_breast_cancer(return_X_y=True)`.

The dataset was split into 70% training data and 30% test data using
stratification and `random_state=42`.

Two baseline models were trained:

- Logistic Regression with `max_iter=1000` and `random_state=42`
- Random Forest with `n_estimators=100` and `random_state=42`

Test accuracy was measured for both models.

For system measurements, one warm-up training run was performed before
five measured training runs. The median training time was reported.

Single-sample inference latency was measured using 100 runs and the
median value was reported.

The models were saved using Joblib. Their file sizes were measured in
bytes and KB. Peak process memory was measured during training and
inference.

## 3. Results

### Accuracy

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 0.9415 |
| Random Forest | 0.9357 |

### System Measurements

| Model | Median Training Time (s) | Median Inference Latency (ms) | Model Size (KB) | Peak Training Memory (MB) | Peak Inference Memory (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.566142 | 0.302000 | 1.03 | 104.50 | 104.51 |
| Random Forest | 0.190324 | 2.610750 | 284.07 | 106.04 | 106.10 |

### Deployment Suitability

The deployment budgets specified in the laboratory instructions were
used to evaluate the two models.

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | Yes | Yes | Yes | No |
| Random Forest | Yes | Yes | Yes | No |

Logistic Regression satisfies the latency and model-size limits for
Cloud, Edge, and Mobile deployment. Its measured memory usage is also
within the maximum memory budget of these platforms.

Random Forest also satisfies the Cloud, Edge, and Mobile latency and
model-size limits. Its measured memory usage is within the maximum
memory budgets for these platforms.

Neither model satisfies the TinyML requirements. Logistic Regression
has a model size below 100 KB and latency below 10 ms, but its measured
memory usage is much greater than the 256 KB TinyML memory limit.
Random Forest exceeds both the 100 KB model-size limit and the 256 KB
memory limit.

## 4. Conclusions

1. Logistic Regression achieved slightly higher test accuracy than
   Random Forest, with accuracies of 0.9415 and 0.9357 respectively.

2. Random Forest trained faster, but Logistic Regression had much
   lower single-sample inference latency and a much smaller model
   size.

3. Both models are suitable for Cloud, Edge, and Mobile deployment
   according to the measured budgets, but neither model satisfies the
   strict TinyML memory requirements.