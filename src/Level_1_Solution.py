"""
================================================================================
README - Level_1_Solution.py
================================================================================
Course:      CSC8370: Data Security (Fall 2026)
Author:      Samuel Robles
Project:     Robust Federated Learning with Attack Detection (MNIST)
Milestone:   Level 1 - Baseline Centralized Convolutional Neural Network (CNN)

Description:
    Implements and trains the baseline centralized Convolutional Neural Network
    (ConvNet) in PyTorch to classify handwritten digits (0-9) from the MNIST
    dataset. This centralized model serves as the architectural foundation and
    performance benchmark prior to expanding into the 10-client Federated
    Learning framework (Level 2) and adding server-side malicious client attack
    detection (Level 3).

Model Architecture (ConvNet):
    - Input:  1 x 28 x 28 grayscale MNIST image tensor (normalized to [-1, 1])
    - Conv1:  Conv2d(1 -> 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
              Output shape: 32 x 14 x 14
    - Conv2:  Conv2d(32 -> 64, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
              Output shape: 64 x 7 x 7
    - FC1:    Linear(64 * 7 * 7 -> 128) + ReLU
    - FC2:    Linear(128 -> 10) output class logits

Data & Training Configuration:
    - Dataset:       MNIST (60,000 train samples / 10,000 test samples)
    - Normalization: Mean=0.5, Std=0.5 (scaled to [-1, 1])
    - Batch Size:    50 (train_loader), 10,000 (test_loader with shuffle=False)
    - Optimizer:     Adam (learning_rate = 0.0005)
    - Loss Function: CrossEntropyLoss
    - Epochs:        5 (default)
    - Seed:          42 (fully reproducible across random, numpy, and torch)
    - Output Model:  Saves best model parameters to 'best_model.pth'

Usage:
    conda activate project
    cd ~/CSC8370_DATA_SECURITY/src

    # 1. Inference-only evaluation (loads best_model.pth directly for Demo Day):
    python Level_1_Solution.py --mode test

    # Optional: specify custom test dataset path and model checkpoint path:
    python Level_1_Solution.py --mode test --data_dir /path/to/eval_data --model_path best_model.pth

    # 2. Centralized training from scratch (trains for 5 epochs and saves best_model.pth):
    python Level_1_Solution.py --mode train
================================================================================
"""

import argparse
import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "mnist")
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pth")


def set_seed(seed=42):
    """
    Set seeds across all random number generators for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataloader(train_dataset, test_dataset):
    """
    Wrap training and test datasets into DataLoaders.
    Note: test_loader uses shuffle=False for deterministic evaluation.
    """
    batch_size = 50

    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        print(f"training has: {len(train_loader)} batch of data!")

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False  # Deterministic order for evaluation
    )
    print(f"testing has: {len(test_loader)} batch of data ({len(test_dataset)} samples)!")
    return train_loader, test_loader


def load_data(data_dir=DEFAULT_DATA_DIR, load_train=True):
    """
    Load MNIST train and test splits with [-1, 1] normalization.
    Allows specifying a custom dataset path via data_dir.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = None
    if load_train:
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        print("The number of training data:", len(train_dataset))

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    print("The number of testing data:", len(test_dataset))

    return dataloader(train_dataset, test_dataset)


class ConvNet(nn.Module):
    """
    Baseline Convolutional Neural Network for MNIST digit classification.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(-1, 64 * 7 * 7)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def evaluate_model(model, test_loader, device):
    """
    Evaluate model classification accuracy on the provided test dataset.
    """
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for test_features, test_labels in test_loader:
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)
            test_pred = model(test_features)
            _, test_indexes = torch.max(test_pred, axis=1)
            num_correct += (test_indexes == test_labels).sum().item()
            num_samples += len(test_labels)

    accuracy = num_correct / num_samples
    return accuracy


def train(epochs=5, learning_rate=0.0005, data_dir=DEFAULT_DATA_DIR, model_save_path=DEFAULT_MODEL_PATH):
    """
    Train the ConvNet model in a centralized manner on MNIST and save the best checkpoint.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting centralized training on device: {device}")
    print(f"Dataset directory: {data_dir}")

    train_loader, test_loader = load_data(data_dir=data_dir, load_train=True)

    model = ConvNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        model.train()

        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            running_acc += (predicted == labels).sum().item() / labels.size(0)

            if i % 100 == 99:
                print(
                    f"Epoch: {epoch + 1:2d} | Batch: {i + 1:4d}/{len(train_loader)} | "
                    f"Loss: {running_loss / 100:.6f} | Acc: {running_acc / 100:.2%}"
                )
                running_loss = 0.0
                running_acc = 0.0

        # Evaluate at epoch end
        val_acc = evaluate_model(model, test_loader, device)
        print(f"[Epoch {epoch + 1} Evaluation] Test Accuracy: {val_acc:.4f} ({val_acc:.2%})")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved new best model to '{model_save_path}' with accuracy: {best_accuracy:.4f}")

    print(f"\nCentralized training complete. Best accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})")
    return best_accuracy


def test(data_dir=DEFAULT_DATA_DIR, model_path=DEFAULT_MODEL_PATH):
    """
    Directly load saved pre-trained model parameters and perform inference-only evaluation.
    This fulfills the Demo Day evaluation requirement for Levels 1 and 2.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference-Only Evaluation on device: {device}")
    print(f"Dataset directory: {data_dir}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint '{model_path}' not found. Please train first with '--mode train'."
        )

    _, test_loader = load_data(data_dir=data_dir, load_train=False)

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded saved model weights from '{model_path}'.")

    accuracy = evaluate_model(model, test_loader, device)
    print("=" * 60)
    print(f"Final Model Test Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("=" * 60)
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="CSC8370 Level 1: Centralized CNN on MNIST (Train & Inference)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test" if os.path.exists(DEFAULT_MODEL_PATH) else "train",
        help="Run 'test' for inference-only using saved weights (Demo Day), or 'train' to train.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to MNIST dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model weights file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate for Adam optimizer (default: 0.0005)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(
            epochs=args.epochs,
            learning_rate=args.lr,
            data_dir=args.data_dir,
            model_save_path=args.model_path
        )
        test(data_dir=args.data_dir, model_path=args.model_path)
    else:
        test(data_dir=args.data_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()
