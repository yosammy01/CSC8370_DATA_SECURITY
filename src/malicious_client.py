"""
================================================================================
malicious_client.py - Level 3 Malicious Client Implementation
================================================================================
Course:      CSC8370: Data Security (Fall 2026)
Author:      Samuel Robles
Project:     Robust Federated Learning with Attack Detection (MNIST)
Milestone:   Level 3 - Malicious Client Update Generation

Description:
    Implements simulated malicious client behaviors to corrupt or degrade
    the global Federated Learning model. Follows the exact function signature
    mandated by the course project instructions so the TA can replace this
    file or function with their own evaluation attack during Demo Day.

Supported Attack Strategies:
    1. 'gaussian_noise': Injects high-variance Gaussian noise into parameter tensors.
    2. 'sign_flip': Inverts parameter signs to push the global model away from minima.
    3. 'zero_out': Zeroes out critical layer weights to cause catastrophic parameter collapse.
================================================================================
"""

import random
import torch


def generate_malicious_update(client_model_state_dict, local_dataset, device, attack_type="random", noise_scale=10.0):
    """
    Args:
        client_model_state_dict (dict): A copy of the current model's state_dict().
        local_dataset (Dataset): The local dataset assigned to this client.
        device (torch.device): CPU or CUDA device.
        attack_type (str): Type of attack. If 'random', dynamically chooses a different
                           attack strategy on each call.
        noise_scale (float): Scale factor for noise injection.
    Returns:
        malicious_state_dict (dict): The modified model parameters (state_dict).
        actual_attack (str): The name of the attack strategy executed.
    """
    available_attacks = ["gaussian_noise", "sign_flip", "zero_out", "scaling_attack", "subtle_drift"]

    if attack_type == "random":
        actual_attack = random.choice(available_attacks)
    else:
        actual_attack = attack_type

    malicious_state_dict = {}

    for key, param in client_model_state_dict.items():
        if not param.is_floating_point():
            malicious_state_dict[key] = param.clone()
            continue

        param_tensor = param.to(device)

        if actual_attack == "gaussian_noise":
            # 1. High-variance random Gaussian noise
            noise = torch.randn_like(param_tensor) * noise_scale
            malicious_state_dict[key] = (param_tensor + noise).cpu()

        elif actual_attack == "sign_flip":
            # 2. Reverses parameter direction to pull global model away from minima
            malicious_state_dict[key] = (-param_tensor * 2.0).cpu()

        elif actual_attack == "zero_out":
            # 3. Complete parameter erasure
            malicious_state_dict[key] = torch.zeros_like(param_tensor).cpu()

        elif actual_attack == "scaling_attack":
            # 4. Multiplies weights by 10x to dominate FedAvg arithmetic mean
            malicious_state_dict[key] = (param_tensor * 10.0).cpu()

        elif actual_attack == "subtle_drift":
            # 5. Low-norm stealthy perturbation targeting directional deviation
            noise = torch.randn_like(param_tensor) * 0.5
            malicious_state_dict[key] = (param_tensor + noise).cpu()

        else:
            # Fallback to standard template Gaussian noise
            malicious_state_dict[key] = (param_tensor + torch.randn_like(param_tensor) * noise_scale).cpu()

    return malicious_state_dict, actual_attack
