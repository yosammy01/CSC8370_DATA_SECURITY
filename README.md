# CSC8370: Data Security — Course Project
**Robust Federated Learning with Attack Detection (MNIST)**  
**Author:** Samuel Robles  
**Semester:** Fall 2026  

---

## 1. Project Overview

The objective of this project is to design and implement a robust Federated Learning (FL) system in PyTorch with 10 independent clients to classify handwritten digits (`0–9`) from the MNIST dataset. The project progresses through three implementation levels:

- **Level 1 (Baseline Centralized CNN):** Build and train a centralized Convolutional Neural Network (`ConvNet`) on the full MNIST dataset to establish the baseline architecture and classification accuracy benchmark.
- **Level 2 (Federated Learning Architecture):** Expand the baseline `ConvNet` to a 10-client Federated Learning architecture where each client trains locally on an independent partition of the MNIST dataset and a central server aggregates parameter updates.
- **Level 3 (Robust Federated Learning with Attack Detection):** Simulate a scenario with 9 benign clients and 1 malicious client (`malicious_client.py`) over 10 global rounds (1 local epoch per round), with a server-side anomaly detection mechanism to flag malicious updates and preserve global model accuracy.

---

## 2. Directory Structure

```text
CSC8370_DATA_SECURITY/
├── README.md                 # Global project documentation
└── src/                      # Source code and trained model artifacts
    ├── Level_1_Solution.py   # Level 1: Baseline centralized CNN implementation
    ├── Level_2_Solution.py   # Level 2: 10-client Federated Learning (FedAvg)
    ├── Level_3_Solution.py   # Level 3: Robust Federated Learning with Attack Detection
    ├── malicious_client.py   # Level 3: Isolated malicious client attack generation
    ├── best_model.pth        # Saved weights of Level 1 centralized CNN (99.02%)
    ├── best_fl_model.pth     # Saved weights of Level 2 federated CNN (98.71%)
    └── data/
        └── mnist/            # Pre-downloaded MNIST dataset (train & test splits)
```

---

## 3. Environment & Dependencies

- **OS:** Linux (x86_64) with NVIDIA CUDA support
- **Conda Environment:** `project` (Python 3.12)
- **Packages:** `torch`, `torchvision`, `numpy`

Activate the environment before running scripts:
```bash
conda activate project
```

---

## 4. Source Files & Execution

### Level 1: Baseline Centralized CNN (`src/Level_1_Solution.py`)
- **Model Architecture (`ConvNet`):**
  - `conv1`: `nn.Conv2d(1, 32, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)`
  - `conv2`: `nn.Conv2d(32, 64, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)`
  - `fc1`: `nn.Linear(64 * 7 * 7, 128)` + `ReLU`
  - `fc2`: `nn.Linear(128, 10)`
- **Training Setup:**
  - Dataset: MNIST (`60,000` training samples, `10,000` test samples), normalized to `[-1, 1]`
  - Batch Size: `50` (training), `10,000` (testing)
  - Optimizer: `Adam` (`lr = 0.0005`)
  - Loss Function: `CrossEntropyLoss`
  - Epochs: `5` (default)
  - Random Seed: `42` (ensures reproducible initialization and data batching)
  - Evaluation: Deterministic test evaluation with `shuffle=False`
- **How to Run:**
  ```bash
  cd ~/CSC8370_DATA_SECURITY/src
  
  # 1. Inference-only test mode (Demo Day evaluation - loads best_model.pth directly):
  python Level_1_Solution.py --mode test

  # Optional: specify custom test dataset path and model checkpoint path:
  python Level_1_Solution.py --mode test --data_dir /path/to/mnist --model_path best_model.pth

  # 2. Centralized training mode (trains for 5 epochs with seed=42 and saves best_model.pth):
  python Level_1_Solution.py --mode train
  ```
- **Baseline Result:** Achieves **99.02%** test accuracy (`0.9902`) after 5 epochs (reproducible with `seed=42`) and saves model weights to `best_model.pth`.

---

### Level 2: 10-Client Federated Learning (`src/Level_2_Solution.py`)
- **Framework & Aggregation:**
  - Standard Federated Averaging (`FedAvg`) aggregating 10 clients.
  - Training dataset partitioned into 10 disjoint subsets of `6,000` samples per client using `random_split`.
- **Training Setup:**
  - Clients: `10`
  - Global Rounds: `10` (fixed by project requirements)
  - Local Epochs: `1` per round (fixed by project requirements)
  - Batch Size: `50` (local client loaders), `1,000` (server test loader)
  - Optimizer: `Adam` (`lr = 0.0005` per client)
  - Random Seed: `42`
- **How to Run:**
  ```bash
  cd ~/CSC8370_DATA_SECURITY/src

  # 1. Inference-only test mode (Demo Day evaluation - loads best_fl_model.pth directly):
  python Level_2_Solution.py --mode test

  # Optional: specify custom test dataset path:
  python Level_2_Solution.py --mode test --data_dir /path/to/mnist --model_path best_fl_model.pth

  # 2. Federated training mode (10 global rounds, 1 local epoch per round):
  python Level_2_Solution.py --mode train
  ```
- **Federated Result:** Achieves **98.71%** global test accuracy (`0.9871`) across 10 clients over 10 global rounds (1 local epoch per round, `seed=42`) and saves model weights to `best_fl_model.pth`.

---

### Level 3: Robust Federated Learning with Attack Detection (`src/Level_3_Solution.py` & `src/malicious_client.py`)
- **System Setup:**
  - 10 total clients: 9 benign clients (`Clients 1–9`) and 1 malicious client (`Client 10`).
  - Strict Demo Day constraint: **10 global communication rounds**, **1 local epoch per round**.
  - **Attack Simulation:** The malicious client trains normally in Round 1 and begins submitting abnormal updates starting in Round 2.
  - **Code Isolation:** Malicious client logic is isolated in `src/malicious_client.py` (`generate_malicious_update(...)`) and marked in `Level_3_Solution.py` with `# === MALICIOUS CLIENT LOGIC HERE ===`.
- **Server Anomaly Detection:**
  - Computes pairwise **Cosine Similarity** and **L2 Norms** of all client parameter update vectors.
  - Flags and quarantines statistical outliers (malicious updates).
  - Performs robust aggregation across verified benign clients only, preserving accuracy.
- **How to Run:**
  ```bash
  cd ~/CSC8370_DATA_SECURITY/src

  # 1. Run live robust FL with attack injection and server-side detection:
  python Level_3_Solution.py --mode train

  # Optional: simulate different attack strategies (gaussian_noise, sign_flip, zero_out):
  python Level_3_Solution.py --mode train --attack_type gaussian_noise
  python Level_3_Solution.py --mode train --attack_type sign_flip

  # 2. Inference-only evaluation of saved robust model:
  python Level_3_Solution.py --mode test
  ```
- **Robust Federated Result:**
  - **Detection Accuracy:** 100% precision and recall (Round 1 flagged None; Rounds 2–10 flagged Client 10 every round with 0 false positives).
  - **Attack-Agnostic Defense:** Validated against dynamic randomized attacks across rounds (`gaussian_noise`, `sign_flip`, `zero_out`, `scaling_attack`, and `subtle_drift`).
  - **Final Model Test Accuracy:** **98.68%** (`0.9868`), completely neutralizing dynamic attacks and preserving clean baseline FL performance.
  - **Saved Weights:** `best_robust_fl_model.pth`.

---

## 5. Project Schedule & Milestones

| Dates | Milestone | Status |
| :--- | :--- | :--- |
| **Oct. 24 – Oct. 30** | Finalize and submit Level 3 project proposal | Completed |
| **Oct. 31 – Nov. 10** | Implement baseline centralized CNN (Level 1) & expand to 10-client FL (Level 2) | Level 1 Completed |
| **Nov. 11 – Nov. 20** | Isolate `malicious_client.py` and implement server-side attack detection (Level 3) | Upcoming |
| **Nov. 21 – Nov. 27** | Finalize `# MALICIOUS CLIENT LOGIC HERE` annotations and submit report & code | Upcoming |
| **Nov. 28 – Dec. 3** | Final verification of detection algorithms and security check | Upcoming |
| **Dec. 4** | Demo Day live training, attack injection, and final accuracy evaluation | Upcoming |
