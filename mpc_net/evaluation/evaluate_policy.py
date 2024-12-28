# mpc_net/evaluation/evaluate_policy.py

import numpy as np
from scipy.spatial import KDTree
import time

def compute_arc_length(xy):
    x = xy[0]
    y = xy[1]
    dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate(([0.0], np.cumsum(dist)))
    return s

def run_learned_policy_on_path(model, path, dynamics,
                               coverage_threshold=0.95,
                               offtrack_dist=2.0):
    """
    Single-step approach, arc coverage. 
    path: shape (2, T).
    """
    # build ref => (3,T)
    yaw_ref = np.arctan2(np.gradient(path[1]), np.gradient(path[0]))
    ref = np.vstack((path, yaw_ref))
    T_path = ref.shape[1]

    arc_array = compute_arc_length(path)
    total_arc = arc_array[-1]

    tree = KDTree(path.T)
    coverage = 0.0
    best_arc = 0.0

    state = np.array([path[0,0], path[1,0], yaw_ref[0]])
    trajectory = [state.copy()]

    cte_list = []
    yaw_list = []
    nn_times = []

    step_count = 0
    max_steps = 2*T_path

    while coverage < coverage_threshold and step_count < max_steps:
        dist, idx = tree.query(state[:2])
        curr_arc = arc_array[idx]
        if curr_arc > best_arc:
            best_arc = curr_arc
        coverage = best_arc / total_arc

        if dist > offtrack_dist:
            print(f"Off track => stop. dist={dist:.2f}")
            break

        ref_x, ref_y, ref_th = ref[0,idx], ref[1,idx], ref[2,idx]
        inp = np.array([[state[0], state[1], state[2],
                         ref_x, ref_y, ref_th]], dtype=np.float32)

        t0 = time.time()
        ctrl = model(inp, training=False).numpy()[0]
        nn_times.append(time.time() - t0)

        # step
        state = dynamics.step(state, ctrl)
        trajectory.append(state.copy())

        cte = np.sqrt((state[0] - ref_x)**2 + (state[1] - ref_y)**2)
        yaw_e = abs(state[2] - ref_th)
        cte_list.append(cte)
        yaw_list.append(yaw_e)

        step_count += 1

    return (np.array(cte_list),
            np.array(yaw_list),
            np.array(nn_times),
            np.array(trajectory))
