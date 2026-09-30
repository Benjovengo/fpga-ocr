#!/usr/bin/env python3

"""
Usage:
    ./python/training.py

Train the floating-point neural network used by the FPGA OCR project.

The MNIST dataset is downloaded to ./data when it is not already available.

The MNIST images are converted to a Kaggle-style CSV representation:

    label,pixel0,pixel1,...,pixel783

Quantization is intentionally not performed in this script.

A separate quantization script will convert the trained floating-point model
into the integer representation required by the FPGA implementation.
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import MNIST


# =============================================================================
# Reproducibility
# =============================================================================

# Set deterministic random seeds for reproducible training runs.
np.random.seed(0)
torch.manual_seed(0)


# =============================================================================
# Files and Folders Structure
# =============================================================================

# training.py is located in:
#
#   <project>/python/training.py
#
# Therefore, parent.parent corresponds to the project root.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Directory containing downloaded and generated training data.
DATA_DIR = PROJECT_DIR / "data"

# Kaggle-style CSV generated from the MNIST training dataset.
TRAIN_FILE = DATA_DIR / "train.csv"

# Directory used to store trained models.
MODEL_DIR = PROJECT_DIR / "model"

# Floating-point trained model.
MODEL_FILE = MODEL_DIR / "mnist_model.pth"


# =============================================================================
# MNIST Dataset
# =============================================================================

def load_mnist():
    """
    Load the MNIST training dataset.

    torchvision checks whether MNIST is already available in DATA_DIR and only
    downloads the dataset when necessary.

    Returns:
        torchvision.datasets.MNIST: MNIST training dataset.
    """

    mnist_downloaded = (
        (DATA_DIR / "MNIST" / "raw" / "train-images-idx3-ubyte").exists()
        and
        (DATA_DIR / "MNIST" / "raw" / "train-labels-idx1-ubyte").exists()
    )

    if mnist_downloaded:
        print(
            f"MNIST dataset already available in: "
            f"{DATA_DIR}"
        )
    else:
        print(
            f"MNIST dataset not found. Downloading to: "
            f"{DATA_DIR}"
        )

    mnist = MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
    )

    print(
        f"MNIST training samples: "
        f"{len(mnist)}"
    )

    return mnist


# =============================================================================
# Convert MNIST to Kaggle-Style CSV
# =============================================================================

def create_training_csv(mnist):
    """
    Convert the torchvision MNIST dataset into the CSV representation expected
    by this project.

    The resulting format is:

        label,pixel0,pixel1,...,pixel783

    Args:
        mnist:
            torchvision MNIST training dataset.
    """

    num_samples = len(mnist)

    images = mnist.data.numpy()
    labels = mnist.targets.numpy()

    # Convert:
    #
    #     N x 28 x 28
    #
    # into:
    #
    #     N x 784
    pixels = images.reshape(
        num_samples,
        784,
    )

    pixel_columns = [
        f"pixel{i}"
        for i in range(784)
    ]

    train_df = pd.DataFrame(
        pixels,
        columns=pixel_columns,
    )

    # The label must be the first CSV column.
    train_df.insert(
        0,
        "label",
        labels,
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    print(f"Created: {TRAIN_FILE}")
    print(f"Samples: {num_samples}")
    print(f"Shape:   {train_df.shape}")
    print(f"Exists:  {TRAIN_FILE.exists()}")


# =============================================================================
# Convert CSV Data to PyTorch Tensors
# =============================================================================

def load_data(filepath):
    """
    Load the Kaggle-style MNIST CSV and convert it to PyTorch tensors.

    Pixel values are binarized:

        pixel <= 127 -> 0
        pixel >  127 -> 1

    Args:
        filepath:
            Path to the training CSV.

    Returns:
        pixels_tensor:
            Floating-point input tensor with shape [samples, 784].

        labels_tensor:
            Integer class-label tensor with shape [samples].
    """

    data = pd.read_csv(filepath)

    labels = data["label"].values

    pixels = data.drop(
        "label",
        axis=1,
    ).values

    # Convert grayscale MNIST pixels into binary input values.
    pixels = (
        pixels > 127
    ).astype(np.float32)

    pixels_tensor = torch.tensor(
        pixels,
        dtype=torch.float32,
    )

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return pixels_tensor, labels_tensor


# =============================================================================
# Split Data into Training and Test Sets
# =============================================================================

def split_data(
    X,
    y,
    train_ratio=0.8,
):
    """
    Split the dataset into training and test datasets.

    A single random permutation is used to guarantee that the training and test
    datasets do not contain overlapping samples.

    Args:
        X:
            Input feature tensor.

        y:
            Label tensor.

        train_ratio:
            Fraction of the complete dataset used for training.

    Returns:
        X_train:
            Training input samples.

        X_test:
            Test input samples.

        y_train:
            Training labels.

        y_test:
            Test labels.
    """

    total_samples = X.shape[0]

    train_size = int(
        total_samples * train_ratio
    )

    indices = torch.randperm(
        total_samples
    )

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    print(
        f"Training data size   : "
        f"{len(X_train)}"
    )

    print(
        f"Training labels size : "
        f"{len(y_train)}"
    )

    print(
        f"Testing data size    : "
        f"{len(X_test)}"
    )

    print(
        f"Testing labels size  : "
        f"{len(y_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =============================================================================
# Linear Layer without Bias
# =============================================================================

class LinearLayer(nn.Module):
    """
    Fully connected neural-network layer without bias.

    Keeping bias disabled simplifies the later FPGA implementation because
    each neuron only requires multiply-accumulate operations on its inputs and
    weights.
    """

    def __init__(
        self,
        in_features,
        out_features,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                in_features,
                out_features,
                dtype=torch.float32,
            )
        )

        # Kaiming initialization is appropriate for layers followed by ReLU.
        nn.init.kaiming_uniform_(
            self.weight,
            nonlinearity="relu",
        )

    def forward(
        self,
        x,
    ):
        return x @ self.weight


# =============================================================================
# Neural Network
# =============================================================================

class NeuralNetwork(nn.Module):
    """
    Fully connected neural network for MNIST digit classification.

    Architecture:

        784 inputs
            |
            v
        Linear 784 -> 64
            |
           ReLU
            |
            v
        Linear 64 -> 64
            |
           ReLU
            |
            v
        Linear 64 -> 32
            |
           ReLU
            |
            v
        Linear 32 -> 10
            |
            v
        logits
    """

    def __init__(
        self,
        input_size=784,
    ):
        super().__init__()

        self.layer1 = LinearLayer(
            input_size,
            64,
        )

        self.layer2 = LinearLayer(
            64,
            64,
        )

        self.layer3 = LinearLayer(
            64,
            32,
        )

        self.layer4 = LinearLayer(
            32,
            10,
        )

    def forward(
        self,
        x,
    ):
        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        x = torch.relu(x)

        x = self.layer4(x)

        return x


# =============================================================================
# Train Neural Network
# =============================================================================

def train_model(
    model,
    X_train,
    y_train,
    epochs=10,
    batch_size=1024,
    learning_rate=0.001,
):
    """
    Train the floating-point neural network.

    Args:
        model:
            Neural network to train.

        X_train:
            Training input samples.

        y_train:
            Training labels.

        epochs:
            Number of complete passes through the training dataset.

        batch_size:
            Number of samples processed per optimization step.

        learning_rate:
            Adam optimizer learning rate.
    """

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    n_samples = X_train.size(0)

    # Ceiling division includes the final incomplete batch.
    n_batches = (n_samples + batch_size - 1) // batch_size

    print()
    print("Training configuration")
    print("----------------------")
    print(f"Training samples : {n_samples}")
    print(f"Batch size       : {batch_size}")
    print(f"Batches/epoch    : {n_batches}")
    print(f"Epochs           : {epochs}")
    print(f"Learning rate    : {learning_rate}")
    print()

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        processed_samples = 0

        # Shuffle the training dataset at the beginning of every epoch.
        indices = torch.randperm(
            n_samples
        )

        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        for batch_index in range(n_batches):
            start_index = (batch_index * batch_size)

            end_index = min(start_index + batch_size, n_samples)

            batch_X = X_epoch[start_index:end_index]

            batch_y = y_epoch[start_index:end_index]

            # -------------------------------------------------------------
            # Forward pass
            # -------------------------------------------------------------

            outputs = model(batch_X)

            loss = criterion(outputs,batch_y)

            # -------------------------------------------------------------
            # Backward pass
            # -------------------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # -------------------------------------------------------------
            # Statistics
            # -------------------------------------------------------------

            total_loss += loss.item()

            predictions = torch.argmax(outputs,dim=1)

            correct += (
                predictions == batch_y
            ).sum().item()

            processed_samples += (
                batch_y.size(0)
            )

        # CrossEntropyLoss returns the mean loss for each batch.
        # Therefore the epoch loss is averaged over the number of batches.
        average_loss = (total_loss/n_batches)

        accuracy = (correct/processed_samples)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )


# =============================================================================
# Evaluate Neural Network
# =============================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the trained model using the test dataset.

    Args:
        model:  Trained neural network.

        X_test: Test input samples.

        y_test: Test labels.

    Returns:
        accuracy: Test-set classification accuracy.
    """

    model.eval()

    with torch.no_grad():
        outputs = model(
            X_test
        )

        predictions = torch.argmax(outputs,dim=1)

        correct = (predictions == y_test).sum().item()

        accuracy = (correct/y_test.size(0))

    print()
    print("Test Results")
    print("------------")

    print(f"Correct classifications : {correct}")
    print(f"Test samples            : {y_test.size(0)}")
    print(f"Test accuracy           : {accuracy:.4f}")

    return accuracy


# =============================================================================
# Save Floating-Point Model
# =============================================================================

def save_model(
    model,
):
    """
    Save the trained floating-point model.

    Quantization is intentionally deferred to a separate script.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_FILE,
    )

    print()
    print(
        f"Saved floating-point model: "
        f"{MODEL_FILE}"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Execute the complete floating-point MNIST training workflow.
    """

    # -------------------------------------------------------------------------
    # Create required directories
    # -------------------------------------------------------------------------

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Working directory : {PROJECT_DIR}")
    print(f"Training file     : {TRAIN_FILE}")
    print(f"Model file        : {MODEL_FILE}")

    # -------------------------------------------------------------------------
    # Download / load MNIST
    # -------------------------------------------------------------------------

    mnist = load_mnist()

    # -------------------------------------------------------------------------
    # Convert MNIST to CSV
    # -------------------------------------------------------------------------

    create_training_csv(mnist)

    # -------------------------------------------------------------------------
    # Load CSV into PyTorch tensors
    # -------------------------------------------------------------------------

    X, y = load_data(TRAIN_FILE)

    print()
    print(f"Total samples  : {X.size(0)}")
    print(f"Input features : {X.size(1)}")
    print(f"Total labels   : {y.size(0)}")

    # -------------------------------------------------------------------------
    # Split dataset (80% for training)
    # -------------------------------------------------------------------------

    (X_train,X_test,y_train,y_test) = split_data(X,y,train_ratio=0.8)

    # -------------------------------------------------------------------------
    # Create neural network
    # -------------------------------------------------------------------------

    model = NeuralNetwork()

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=10,
        batch_size=1024,
        learning_rate=0.001,
    )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    evaluate_model(model=model,X_test=X_test,y_test=y_test)

    # -------------------------------------------------------------------------
    # Save floating-point model
    # -------------------------------------------------------------------------

    save_model(model)

    # Check min and max values of the trained floating-point weights.
    print("Floating-point weights:")

    for layer in model.children():
        if isinstance(layer, LinearLayer):
            print(
                f"Layer weights min: {layer.weight.min().item():.6f}, "
                f"max: {layer.weight.max().item():.6f}"
            )

# =============================================================================
# Program Entry Point
# =============================================================================

if __name__ == "__main__":
    main()