#!/usr/bin/env python3

print("evaluatin begins")
"""
scripts/evaluate.py

Load 'moe_model.keras' and run it on the bigger figure-8 path
with coverage_threshold=0.99, max_steps=3*T => better chance to finish.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

    # build a bigger fig8 path
    path = vertical_fig8(T=4000, a=10, b=5)  # same as in generate__paths

    # unicycle
    dynamics = UnicycleDynamics(dt=0.01)

    # run policy, with coverage_threshold=0.99, offtrack_dist=2.0
    # see evaluate_policy.py => you can pass these as args, or
    # you can open that function & set max_steps=3*T
    cte, yaw_err, times, traj = run_learned_policy_on_path(
        model=model,
        path=path,
        dynamics=dynamics,
        coverage_threshold=0.99,
        offtrack_dist=2.0
    )

    print(f"Policy steps={len(cte)}, avg CTE={np.mean(cte):.3f}, avg Yaw={np.mean(yaw_err):.3f}")

    # Plot error
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
    if len(traj)>0:
        plt.plot(traj[:,0], traj[:,1], 'r-', label='Policy')
    plt.axis('equal')
    plt.legend()
    plt.title("Learned Policy on Large Fig-8")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()
