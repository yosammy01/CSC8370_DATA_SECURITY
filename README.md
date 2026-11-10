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
    ├── best_model.pth        # Saved weights of best Level 1 centralized CNN
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
  - Epochs: `3`
- **How to Run:**
  ```bash
  cd ~/CSC8370_DATA_SECURITY/src
  
  # 1. Inference-only test mode (Demo Day evaluation - loads best_model.pth directly):
  python Level_1_Solution.py --mode test

  # Optional: specify custom test dataset path and model checkpoint path:
  python Level_1_Solution.py --mode test --data_dir /path/to/mnist --model_path best_model.pth

  # 2. Centralized training mode (trains for 3 epochs and saves best_model.pth):
  python Level_1_Solution.py --mode train
  ```
- **Baseline Result:** Achieves **99.04%** test accuracy (`0.9904`) after 3 epochs and saves model weights to `best_model.pth`.

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
