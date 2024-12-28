#!/usr/bin/env python3

print("DEBUG: Entering train.py")

"""
scripts/train.py

Collect data from  paths, train a bigger MoE model,
save as 'moe_model.keras'.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from mpc_net.dynamics.unicycle import UnicycleDynamics
from mpc_net.paths.path_generators import generate__paths
from mpc_net.training.data_collection import collect_mpc_data
from mpc_net.training.train_moe import train_mixture_of_experts
from mpc_net.models.mixture_of_experts import MixtureOfExperts

def main():
    # 1) Setup constraints & dynamics
    constraints = {
        'v_min': 0.0,
        'v_max': 2.0,
        'omega_min': -1.5,
        'omega_max': 1.5
    }
    dynamics = UnicycleDynamics(dt=0.01)

    # 2) Generate  paths
    path_gens = generate__paths()

    # 3) Collect MPC data (using decimation=5 for speed)
    X_data, Y_data, mpc_trajs = collect_mpc_data(
        path_generators=path_gens,
        constraints=constraints,
        dynamics=dynamics,
        coverage_threshold=0.99,
        replay_buffer_size=300000,
        decimation=5
    )

    print(f"\nCollected {len(X_data)} data points: X shape={X_data.shape}, Y={Y_data.shape}")

    # 4) Plot coverage
    plt.figure()
    for name, traj in mpc_trajs.items():
        if len(traj) > 0:
            plt.plot(traj[:,0], traj[:,1], label=name)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("MPC-Collected Trajectories on  Paths")
    plt.show()

    # 5) Train or Load
    model_path = "moe_model.keras"
    abs_path   = os.path.abspath(model_path)
    if os.path.exists(abs_path):
        print(f"Loading existing model from {abs_path}")
        model = tf.keras.models.load_model(
            abs_path,
            custom_objects={'MixtureOfExperts': MixtureOfExperts},
            compile=False
        )
        print("\nDEBUG: Model summary after load from disk:")
        model.summary()
    else:
        print("\nNo existing model found. Training new MoE model...")
        model, history = train_mixture_of_experts(
            X_data, Y_data,
            num_experts=12,  # bigger
            hidden_dim=64,   # bigger
            lr=1e-3,
            batch_size=32,
            epochs=300       # train longer
        )

        print("DEBUG: Model summary post-training:")
        model.summary()

        # Save in .keras format
        print(f"\nSaving new model to: {abs_path}")
        model.save(abs_path)
        print("DEBUG: Model saved.")

        # Immediately reload to confirm
        reloaded = tf.keras.models.load_model(
            abs_path,
            custom_objects={'MixtureOfExperts': MixtureOfExperts},
            compile=False
        )
        print("DEBUG: Reloaded model summary =>")
        reloaded.summary()

    print("\nDone training. Model is available at:", abs_path)

if __name__ == "__main__":
    main()
