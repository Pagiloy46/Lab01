import sys

import numpy
import pandas
import sklearn
import scipy
import matplotlib
import seaborn
import torch
import torchvision
import torchinfo
import thop
import onnx
import onnxruntime
import mlflow
import memory_profiler
import psutil
import codecarbon
import fastapi
import uvicorn
import pytest
import httpx
import locust
import requests
import pyarrow
import joblib
import tqdm


print(f"Python: {sys.version}")

packages = {
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": seaborn.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchinfo": torchinfo.__version__,
    "thop": thop.__version__,
    "onnx": onnx.__version__,
    "onnxruntime": onnxruntime.__version__,
    "mlflow": mlflow.__version__,
    "memory-profiler": memory_profiler.__version__,
    "psutil": psutil.__version__,
    "codecarbon": codecarbon.__version__,
    "fastapi": fastapi.__version__,
    "uvicorn": uvicorn.__version__,
    "pytest": pytest.__version__,
    "httpx": httpx.__version__,
    "locust": locust.__version__,
    "requests": requests.__version__,
    "pyarrow": pyarrow.__version__,
    "joblib": joblib.__version__,
    "tqdm": tqdm.__version__,
}

for name, version in packages.items():
    print(f"{name}: {version}")