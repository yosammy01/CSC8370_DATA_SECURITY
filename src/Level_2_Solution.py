"""
================================================================================
README - Level_2_Solution.py
================================================================================
Course:      CSC8370: Data Security (Fall 2026)
Author:      Samuel Robles
Project:     Robust Federated Learning with Attack Detection (MNIST)
Milestone:   Level 2 - 10-Client Federated Learning Architecture (FedAvg)

Description:
    Expands the baseline centralized Convolutional Neural Network (ConvNet) from
    Level 1 into a 10-client Federated Learning (FL) system using Federated
    Averaging (FedAvg), based on Dong Yang's course framework template.
    The 60,000 MNIST training images are partitioned into 10 disjoint client
    datasets (6,000 samples each) using random_split. Across exactly 10 global
    rounds, each client trains for a fixed 1 local epoch per round, after which
    the central server averages the client model parameters to update the global
    model.

Model Architecture (ConvNet):
    - Input:  1 x 28 x 28 grayscale MNIST image tensor (normalized to [-1, 1])
    - Conv1:  Conv2d(1 -> 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
              Output shape: 32 x 14 x 14
    - Conv2:  Conv2d(32 -> 64, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
              Output shape: 64 x 7 x 7
    - FC1:    Linear(64 * 7 * 7 -> 128) + ReLU
    - FC2:    Linear(128 -> 10) output class logits

Federated Training Configuration:
    - Dataset:        MNIST (60,000 train samples split across 10 clients / 10,000 test)
    - Clients:        10 independent clients (6,000 samples / 120 batches per client)
    - Batch Size:     50 (client train_loaders), 1,000 (test_loader with shuffle=False)
    - Optimizer:      Adam (learning_rate = 0.0005 per client)
    - Loss Function:  CrossEntropyLoss
    - Global Rounds:  10 (as strictly stipulated by course requirements)
    - Local Epochs:   1 per round (fixed)
    - Aggregation:    Federated Averaging (FedAvg)
    - Output Model:   Saves best global model parameters to 'best_fl_model.pth'
                      (and final model to 'federated_model.pth')

Usage:
    conda activate project
    cd ~/CSC8370_DATA_SECURITY/src

    # 1. Inference-only evaluation (loads saved best_fl_model.pth for Demo Day):
    python Level_2_Solution.py --mode test

    # Optional: specify custom test dataset path and model checkpoint path:
    python Level_2_Solution.py --mode test --data_dir /path/to/eval_data --model_path best_fl_model.pth

    # 2. Federated training from scratch (10 rounds, 1 local epoch per round):
    python Level_2_Solution.py --mode train
================================================================================
"""

import argparse
import copy
import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "mnist")
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_fl_model.pth")


def set_seed(seed=42):
    """
    Set seeds across all random number generators for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# CNN model definition, identical to the model in Level 1
class ConvNet(nn.Module):
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


# Load data (each client will load its own data in a real FL scenario)
def load_data(transform, data_dir=DEFAULT_DATA_DIR, datasets="MNIST"):
    if datasets == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=False, download=True, transform=transform
        )

    return train_dataset, test_dataset


# Split the dataset into 'n_clients' partitions
def partition_dataset(dataset, n_clients=10):
    split_size = len(dataset) // n_clients
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [split_size] * n_clients, generator=generator)


# Client-side local training
def client_update(client_model, optimizer, train_loader, device, epochs=1):
    """
    Train client model on local partition for 'epochs' local epoch(s).
    """
    client_model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            output = client_model(data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


# Server-side aggregation of client models (FedAvg)
def server_aggregate(global_model, client_models):
    """
    Federated Averaging (FedAvg): Averages the parameter tensors across all
    client models, updates global_model, and synchronizes client models.
    """
    global_dict = global_model.state_dict()
    n_clients = len(client_models)

    for key in global_dict.keys():
        if global_dict[key].is_floating_point():
            # Compute element-wise mean across all client models
            stacked = torch.stack([client.state_dict()[key].float() for client in client_models], dim=0)
            global_dict[key] = torch.mean(stacked, dim=0)
        else:
            # Non-floating point tensors (e.g. num_batches_tracked)
            global_dict[key] = client_models[0].state_dict()[key]

    global_model.load_state_dict(global_dict)

    # Distribute updated global weights back to all client models
    for client in client_models:
        client.load_state_dict(copy.deepcopy(global_dict))


# Test model on test dataset
def test_model(model, test_loader, device):
    """
    Evaluate model classification accuracy on test dataset.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Federated Learning process
def federated_learning(n_clients=10, global_epochs=10, local_epochs=1,
                       data_dir=DEFAULT_DATA_DIR, model_save_path=DEFAULT_MODEL_PATH):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Federated Learning on device: {device}")
    print(f"Clients: {n_clients} | Global Rounds: {global_epochs} | Local Epochs: {local_epochs}")
    print(f"Dataset directory: {data_dir}")

    # Set up the data transformation and load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset, test_dataset = load_data(transform, data_dir=data_dir)

    # Partition the dataset for each client
    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [
        DataLoader(dataset, batch_size=50, shuffle=True) for dataset in client_datasets
    ]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"Partitioned {len(train_dataset)} samples into {n_clients} client partitions "
          f"({len(train_dataset) // n_clients} samples / {len(client_loaders[0])} batches each).")
    print(f"Test dataset has {len(test_dataset)} samples ({len(test_loader)} batches).")

    # Initialize global model and n_clients client models
    global_model = ConvNet().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]

    # Optimizers for client models
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.0005) for model in client_models
    ]

    best_accuracy = 0.0

    # Federated Learning process
    for global_epoch in range(global_epochs):
        print(f"\n--- Global Round {global_epoch + 1}/{global_epochs} ---")

        # Each client trains locally
        for client_idx in range(n_clients):
            loss, acc = client_update(
                client_models[client_idx],
                optimizers[client_idx],
                client_loaders[client_idx],
                device,
                local_epochs,
            )
            print(f"  Client {client_idx + 1:2d}/{n_clients} | Local Loss: {loss:.4f} | Local Acc: {acc:.2%}")

        # Server aggregates the models
        server_aggregate(global_model, client_models)

        # Evaluate global model on test dataset
        test_accuracy = test_model(global_model, test_loader, device)
        print(f"[Server] Global Model Test Accuracy after round {global_epoch + 1}: {test_accuracy:.4f} ({test_accuracy:.2%})")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(global_model.state_dict(), model_save_path)
            print(f"--> Saved new best global model to '{model_save_path}' (Accuracy: {best_accuracy:.4f})")

    # Also save final model weights to 'federated_model.pth' as in template
    final_path = os.path.join(SCRIPT_DIR, "federated_model.pth")
    torch.save(global_model.state_dict(), final_path)
    print(f"\nSaved final global model to '{final_path}'.")
    print(f"Best Global FL Test Accuracy achieved: {best_accuracy:.4f} ({best_accuracy:.2%})")
    print("Federated learning process completed.")
    return best_accuracy


# Direct inference-only evaluation for Demo Day
def run_test_only(data_dir=DEFAULT_DATA_DIR, model_path=DEFAULT_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference-Only Evaluation on device: {device}")
    print(f"Dataset directory: {data_dir}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint '{model_path}' not found. Please train first with '--mode train'."
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    _, test_dataset = load_data(transform, data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded saved global model weights from '{model_path}'.")

    accuracy = test_model(model, test_loader, device)
    print("=" * 60)
    print(f"Final Global Model Test Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("=" * 60)
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="CSC8370 Level 2: 10-Client Federated Learning on MNIST"
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
        help=f"Path to model checkpoint file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--global_epochs",
        type=int,
        default=10,
        help="Number of global communication rounds (default: 10)",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local epochs per round (default: 1, stipulated by requirements)",
    )
    parser.add_argument(
        "--n_clients",
        type=int,
        default=10,
        help="Number of federated clients (default: 10)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        federated_learning(
            n_clients=args.n_clients,
            global_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            data_dir=args.data_dir,
            model_save_path=args.model_path,
        )
        run_test_only(data_dir=args.data_dir, model_path=args.model_path)
    else:
        run_test_only(data_dir=args.data_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()
