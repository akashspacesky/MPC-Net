#!/usr/bin/env python3

print("evaluatin begins")

"""
scripts/evaluate.py

1) Loads 'moe_model.keras'
2) Runs it on a "gentle" figure-8 path (from mpc_net.paths) 
3) Plots cross-track & yaw error, final trajectory
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Our custom modules
from mpc_net.dynamics.unicycle import UnicycleDynamics
from mpc_net.evaluation.evaluate_policy import run_learned_policy_on_path
from mpc_net.paths.path_generators import vertical_fig8
from mpc_net.models.mixture_of_experts import MixtureOfExperts

def main():
    model_path = "moe_model.keras"
    abs_path   = os.path.abspath(model_path)

    print("DEBUG: Attempting to load model from:", abs_path)
    if not os.path.exists(abs_path):
        print(f"No model found at {abs_path}. Please train first.")
        return

    # load model
    model = tf.keras.models.load_model(
        abs_path,
        custom_objects={'MixtureOfExperts': MixtureOfExperts},
        compile=False
    )
    print("\nDEBUG: Loaded model summary =>")
    model.summary()

    # build a path
    path = vertical_fig8(T=2000, a=8, b=3)

    # unicycle
    dynamics = UnicycleDynamics(dt=0.01)

    # run policy
    cte, yaw_err, times, traj = run_learned_policy_on_path(
        model=model,
        path=path,
        dynamics=dynamics,
        coverage_threshold=0.95,
        offtrack_dist=2.0
    )

    print(f"Policy steps={len(cte)}, avg CTE={np.mean(cte):.3f}, avg Yaw={np.mean(yaw_err):.3f}")

    # Plot error over steps
    fig, axs = plt.subplots(2,1, figsize=(8,8))
    axs[0].plot(cte, label='CTE')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_ylabel("CTE (m)")

    axs[1].plot(yaw_err, label='Yaw Err')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Yaw Error (rad)")

    plt.tight_layout()
    plt.show()

    # plot final trajectory
    plt.figure()
    plt.plot(path[0], path[1], 'k--', label='Ref Path')
    if len(traj) > 0:
        plt.plot(traj[:,0], traj[:,1], 'r-', label='Policy')
    plt.axis('equal')
    plt.legend()
    plt.title("Learned Policy on Gentle Fig-8")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()
