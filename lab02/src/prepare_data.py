
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
X, y = load_digits(return_X_y=True)

# Scale pixel values
X = X.astype(np.float64) / 16.0

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# Split dataset: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_one_hot,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

print("Dataset prepared successfully!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("Number of classes:", y_one_hot.shape[1])
print("Minimum input value:", X.min())
print("Maximum input value:", X.max())
