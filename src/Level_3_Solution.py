"""
================================================================================
README - Level_3_Solution.py
================================================================================
Course:      CSC8370: Data Security (Fall 2026)
Author:      Samuel Robles
Project:     Robust Federated Learning with Attack Detection (MNIST)
Milestone:   Level 3 - Robust Federated Learning with Attack Detection

Description:
    Implements a 10-client Federated Learning system with robust server-side
    attack detection under adversarial conditions. The system features 9 benign
    clients and 1 malicious client. In Round 1, all clients train normally.
    Starting from Round 2 onward, the malicious client uploads abnormal updates
    generated via malicious_client.py. The central server applies a multi-metric
    anomaly detection algorithm (cosine similarity and L2 norm outlier detection)
    to identify, flag, and filter out malicious updates, preserving global
    model accuracy (~98.7%) across all 10 rounds.

Model Architecture (ConvNet):
    - Input:  1 x 28 x 28 grayscale MNIST image tensor (normalized to [-1, 1])
    - Conv1:  Conv2d(1 -> 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
              Output shape: 32 x 14 x 14
    - Conv2:  Conv2d(32 -> 64, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
              Output shape: 64 x 7 x 7
    - FC1:    Linear(64 * 7 * 7 -> 128) + ReLU
    - FC2:    Linear(128 -> 10) output class logits

Federated Training & Defense Configuration:
    - Dataset:        MNIST (60,000 train samples partitioned across 10 clients / 10,000 test)
    - Clients:        10 total (9 Benign: Clients 1-9; 1 Malicious: Client 10)
    - Global Rounds:  10 (strictly stipulated)
    - Local Epochs:   1 per round (strictly stipulated)
    - Malicious Mode: Normal in Round 1; submits abnormal updates starting Round 2
    - Defense:        Server-side cosine similarity clustering and L2 norm anomaly detection
    - Output Model:   Saves best robust global model parameters to 'best_robust_fl_model.pth'

Usage:
    conda activate project
    cd ~/CSC8370_DATA_SECURITY/src

    # 1. Run live robust federated learning with attack detection:
    python Level_3_Solution.py --mode train

    # Optional: customize attack strategy or malicious client ID (0-indexed):
    python Level_3_Solution.py --mode train --attack_type gaussian_noise
    python Level_3_Solution.py --mode train --malicious_id 9

    # 2. Inference-only evaluation of saved robust model:
    python Level_3_Solution.py --mode test
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

# Import the isolated malicious client logic required for Demo Day
from malicious_client import generate_malicious_update

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "mnist")
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_robust_fl_model.pth")


def set_seed(seed=42):
    """
    Set seeds across all random number generators for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_data(transform, data_dir=DEFAULT_DATA_DIR):
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def partition_dataset(dataset, n_clients=10):
    split_size = len(dataset) // n_clients
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [split_size] * n_clients, generator=generator)


def client_update(client_model, optimizer, train_loader, device, epochs=1):
    """
    Standard benign client training on local partition for specified epochs.
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


def flatten_model_update(client_state, global_state):
    """
    Compute delta vector (W_client - W_global) and flatten into a 1D tensor.
    """
    diff_tensors = []
    for key in global_state.keys():
        if global_state[key].is_floating_point():
            diff = (client_state[key] - global_state[key]).float().view(-1)
            diff_tensors.append(diff)
    return torch.cat(diff_tensors)


def detect_malicious_clients(client_models, global_model):
    """
    Server-side anomaly detection mechanism using Distance-to-Median (Multi-Krum style):
    1. Computes update vector deltas: Delta_k = W_k - W_global for each client.
    2. Constructs a robust reference update: Delta_median = median(Delta_1, ..., Delta_K).
       Since 9 of 10 clients are benign, the coordinate-wise median is resilient to outliers.
    3. Computes the Euclidean distance of each client to Delta_median: d_i = ||Delta_i - Delta_median||_2.
    4. Benign clients cluster closely around the median (small d_i).
       Malicious updates (noise, sign-flip, large perturbations) produce massive d_i.
    5. Flags clients whose distance to median significantly exceeds the cluster distribution.
    Returns:
        flagged_clients (list): List of client indices detected as malicious.
        scores (dict): Diagnostic metrics for each client.
    """
    global_state = global_model.state_dict()
    n_clients = len(client_models)

    # Flatten update vectors for all clients
    deltas = []
    for client in client_models:
        delta = flatten_model_update(client.state_dict(), global_state)
        deltas.append(delta)
    deltas_tensor = torch.stack(deltas, dim=0)  # Shape: [n_clients, num_params]

    # Coordinate-wise median of all client updates (robust baseline)
    median_delta = torch.median(deltas_tensor, dim=0).values

    # Euclidean distance of each client update to the robust median
    distances = torch.norm(deltas_tensor - median_delta.unsqueeze(0), p=2, dim=1).cpu().numpy()
    
    # Also calculate pairwise cosine similarity to median update for diagnostics
    norm_deltas = torch.nn.functional.normalize(deltas_tensor, p=2, dim=1)
    norm_median = torch.nn.functional.normalize(median_delta.unsqueeze(0), p=2, dim=1)
    cos_to_median = torch.mm(norm_deltas, norm_median.t()).squeeze(1).cpu().numpy()

    # Median and threshold of distances
    med_dist = np.median(distances)
    # Distance threshold: an update is an anomaly if its distance is more than 3.5x the median distance
    # with an absolute safety margin of at least 5.0
    dist_threshold = max(3.5 * med_dist, med_dist + 5.0)

    # Consensus directional alignment across clients
    med_cos = np.median(cos_to_median)

    flagged_clients = []
    scores = {}

    for i in range(n_clients):
        d_val = float(distances[i])
        cos_val = float(cos_to_median[i])
        
        # Anomaly conditions covering all four adversarial attack categories:
        # 1. Magnitude Outlier: Noise injection, parameter scaling, zero-out attacks
        is_distance_anomaly = d_val > dist_threshold

        # 2. Directional Inversion: Sign-flipping, gradient negation, orthogonal noise
        is_direction_anomaly = cos_val < -0.10

        # 3. Stealthy / Subtle Deviation: Targeted poisoning or backdoor that keeps norm small
        #    but significantly deviates from the benign 9-client consensus direction
        is_stealthy_anomaly = (cos_val < 0.20) or (cos_val < med_cos - 0.40 and d_val > 1.5 * med_dist)

        is_malicious = is_distance_anomaly or is_direction_anomaly or is_stealthy_anomaly
        if is_malicious:
            flagged_clients.append(i)

        scores[i] = {
            "distance_to_median": d_val,
            "cos_to_median": cos_val,
            "is_anomaly": is_malicious
        }

    return flagged_clients, scores


def robust_server_aggregate(global_model, client_models, flagged_clients):
    """
    Aggregates only the benign client models (filtering out flagged malicious clients).
    Synchronizes the clean aggregated weights back to all clients.
    """
    n_clients = len(client_models)
    benign_indices = [i for i in range(n_clients) if i not in flagged_clients]

    if len(benign_indices) == 0:
        print("[WARNING] All clients flagged! Retaining previous global model.")
        return

    global_dict = global_model.state_dict()
    benign_models = [client_models[i] for i in benign_indices]

    for key in global_dict.keys():
        if global_dict[key].is_floating_point():
            stacked = torch.stack([client.state_dict()[key].float() for client in benign_models], dim=0)
            global_dict[key] = torch.mean(stacked, dim=0)
        else:
            global_dict[key] = benign_models[0].state_dict()[key]

    global_model.load_state_dict(global_dict)

    # Synchronize cleaned global weights to all clients
    for client in client_models:
        client.load_state_dict(copy.deepcopy(global_dict))


def test_model(model, test_loader, device):
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


def robust_federated_learning(n_clients=10, global_epochs=10, local_epochs=1,
                              malicious_client_id=9, attack_type="gaussian_noise",
                              data_dir=DEFAULT_DATA_DIR, model_save_path=DEFAULT_MODEL_PATH):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("CSC8370 LEVEL 3: ROBUST FEDERATED LEARNING WITH ATTACK DETECTION")
    print(f"Device: {device} | Total Clients: {n_clients} | Global Rounds: {global_epochs}")
    print(f"Benign Clients: 0-{n_clients-2} (1-{n_clients-1}) | Malicious Client: {malicious_client_id} (Client {malicious_client_id+1})")
    print(f"Simulated Attack Strategy: '{attack_type}' (starting at Round 2)")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset, test_dataset = load_data(transform, data_dir=data_dir)

    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [
        DataLoader(dataset, batch_size=50, shuffle=True) for dataset in client_datasets
    ]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    global_model = ConvNet().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.0005) for model in client_models]

    best_accuracy = 0.0
    detection_history = []

    for global_epoch in range(global_epochs):
        round_num = global_epoch + 1
        print(f"\n>>> Global Communication Round {round_num}/{global_epochs} <<<")

        # -------------------------------------------------------------
        # 1. CLIENT LOCAL TRAINING PHASE
        # -------------------------------------------------------------
        for client_idx in range(n_clients):
            if client_idx == malicious_client_id and round_num >= 2:
                # =============================================================
                # === MALICIOUS CLIENT LOGIC HERE ===
                # (Required by Course Instructions for TA live Demo Day testing)
                # =============================================================
                try:
                    # Supports 4 arguments (our dynamic testing module)
                    malicious_result = generate_malicious_update(
                        client_model_state_dict=copy.deepcopy(client_models[client_idx].state_dict()),
                        local_dataset=client_datasets[client_idx],
                        device=device,
                        attack_type=attack_type
                    )
                except TypeError:
                    # Fallback to standard 3-argument signature in TA's course template:
                    # generate_malicious_update(client_model_state_dict, local_dataset, device)
                    malicious_result = generate_malicious_update(
                        client_model_state_dict=copy.deepcopy(client_models[client_idx].state_dict()),
                        local_dataset=client_datasets[client_idx],
                        device=device
                    )

                if isinstance(malicious_result, tuple):
                    malicious_state, chosen_attack = malicious_result
                    print(f"  [ATTACK INJECTED] Client {client_idx + 1} (Malicious) injected dynamic attack: '{chosen_attack}'")
                else:
                    malicious_state = malicious_result
                    print(f"  [ATTACK INJECTED] Client {client_idx + 1} (Malicious) injected malicious model update.")

                # Ensure all returned parameter tensors match the active compute device (CPU or CUDA)
                malicious_state = {k: v.to(device) for k, v in malicious_state.items()}
                client_models[client_idx].load_state_dict(malicious_state)
            else:
                # Benign local training
                loss, acc = client_update(
                    client_models[client_idx],
                    optimizers[client_idx],
                    client_loaders[client_idx],
                    device,
                    local_epochs
                )
                role = "Benign" if client_idx != malicious_client_id else "Malicious (Honest in R1)"
                print(f"  Client {client_idx + 1:2d} ({role:20s}) | Loss: {loss:.4f} | Acc: {acc:.2%}")

        # -------------------------------------------------------------
        # 2. SERVER-SIDE ANOMALY DETECTION PHASE
        # -------------------------------------------------------------
        flagged_clients, scores = detect_malicious_clients(client_models, global_model)

        if len(flagged_clients) > 0:
            flagged_display = [f"Client {c + 1}" for c in flagged_clients]
            print(f"\n  [ALERT] Server Anomaly Detector flagged {len(flagged_clients)} client(s): {', '.join(flagged_display)}")
            for c in flagged_clients:
                print(f"          - Client {c + 1}: Dist to Median = {scores[c]['distance_to_median']:.4f}, Cos to Median = {scores[c]['cos_to_median']:.4f}")
        else:
            print("  [INFO] Server Anomaly Detector: All client updates verified benign.")

        detection_history.append((round_num, flagged_clients))

        # -------------------------------------------------------------
        # 3. ROBUST AGGREGATION & GLOBAL EVALUATION
        # -------------------------------------------------------------
        robust_server_aggregate(global_model, client_models, flagged_clients)
        test_accuracy = test_model(global_model, test_loader, device)
        print(f"  [Server] Robust Global Test Accuracy after Round {round_num}: {test_accuracy:.4f} ({test_accuracy:.2%})")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(global_model.state_dict(), model_save_path)
            print(f"  --> Saved new best robust model to '{model_save_path}' (Accuracy: {best_accuracy:.4f})")

    # Final summary
    print("\n" + "=" * 70)
    print("ROBUST FEDERATED LEARNING DEMO SUMMARY")
    print("=" * 70)
    print("Round-by-Round Malicious Client Detection Record:")
    for r, flagged in detection_history:
        flagged_str = f"Client {[c + 1 for c in flagged]}" if flagged else "None (All Benign)"
        print(f"  - Round {r:2d}: Flagged {flagged_str}")

    print(f"\nFinal Best Robust Model Test Accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})")
    print(f"Saved robust checkpoint: '{model_save_path}'")
    return best_accuracy


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
    train_dataset, test_dataset = load_data(transform, data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded saved robust global model weights from '{model_path}'.")

    accuracy = test_model(model, test_loader, device)
    print("=" * 60)
    print(f"Final Robust Global Model Test Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("=" * 60)
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="CSC8370 Level 3: Robust Federated Learning with Attack Detection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Run 'train' for live FL simulation with attack & detection, or 'test' for inference evaluation.",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="random",
        help="Simulated attack strategy for Client 10 ('random', 'gaussian_noise', 'sign_flip', 'zero_out', 'scaling_attack', 'subtle_drift')",
    )
    parser.add_argument(
        "--malicious_id",
        type=int,
        default=9,
        help="0-indexed ID of the malicious client (default: 9 for Client 10)",
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

    args = parser.parse_args()

    if args.mode == "train":
        robust_federated_learning(
            n_clients=10,
            global_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            malicious_client_id=args.malicious_id,
            attack_type=args.attack_type,
            data_dir=args.data_dir,
            model_save_path=args.model_path,
        )
    else:
        run_test_only(data_dir=args.data_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()
