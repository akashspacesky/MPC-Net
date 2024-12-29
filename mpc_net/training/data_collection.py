"""
mpc_net/training/data_collection.py

Collect single-step MPC data with arc-length coverage.
Now includes:
  - coverage_threshold=0.95
  - max_steps=2*T_path
  - decimation=5 by default
  - 'stale coverage' detection to avoid infinite loops
"""

import numpy as np
from scipy.spatial import KDTree
from mpc_net.mpc.solver import solve_mpc

def compute_arc_length(xy):
    x = xy[0]
    y = xy[1]
    dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate(([0.0], np.cumsum(dist)))
    return s

def collect_mpc_data(path_generators,
                     constraints,
                     dynamics,
                     coverage_threshold=0.95,
                     replay_buffer_size=30000,
                     decimation=5):
    """
    Single-step approach w/ arc coverage:
      - coverage_threshold=0.95
      - decimation=5 => skip storing some data
      - max_steps=2*T_path => ensures we don't loop forever
      - stale coverage detection => break if coverage not improving for 100 steps
    """
    X_data = []
    Y_data = []
    mpc_trajectories = {}

    for name, path_func in path_generators:
        print(f"\n[MPC Data] Running path: {name}")
        path = path_func()  # shape(2,T)
        yaw_ref = np.arctan2(np.gradient(path[1]), np.gradient(path[0]))
        ref_path = np.vstack((path, yaw_ref))  # shape(3, T)
        T_path = ref_path.shape[1]

        # arc-based coverage
        arc_array = compute_arc_length(path)
        total_arc = arc_array[-1]

        collect_state = np.array([path[0,0], path[1,0], yaw_ref[0]])
        tree = KDTree(path.T)

        local_traj = [collect_state.copy()]
        best_arc = 0.0
        coverage = 0.0

        iteration_count = 0
        max_steps = 2 * T_path  # 2x path length => limit
        step_counter = 0

        stale_count = 0
        last_coverage = 0.0

        while (coverage < coverage_threshold
               and iteration_count < max_steps
               and len(X_data) < replay_buffer_size):

            dist, idx = tree.query(collect_state[:2])
            curr_arc = arc_array[idx]
            if curr_arc > best_arc:
                best_arc = curr_arc
            coverage = best_arc / total_arc

            # stale coverage check
            coverage_diff = abs(coverage - last_coverage)
            if coverage_diff < 1e-7:
                stale_count += 1
            else:
                stale_count = 0
            last_coverage = coverage

            # break if coverage is stuck not improving
            if stale_count > 100:
                print("Coverage not improving => break early.")
                break

            if dist > 2.0:
                print(f"  -> Off track => stop. dist={dist:.2f}")
                break

            # local horizon slice
            horizon_steps = 50
            start_idx = max(0, idx)
            end_idx = min(T_path, start_idx + horizon_steps)
            ref_horizon = ref_path[:, start_idx:end_idx]

            # Solve MPC
            u_sol, _ = solve_mpc(collect_state, ref_horizon, constraints)
            if u_sol.ndim < 2 or u_sol.size == 0:
                print(f"  -> MPC solution empty => stop.")
                break

            ctrl = u_sol[:, 0]

            # decimate => store every Nth step
            if step_counter % decimation == 0:
                ref_x  = ref_horizon[0,0]
                ref_y  = ref_horizon[1,0]
                ref_th = ref_horizon[2,0]
                X_data.append([
                    collect_state[0], collect_state[1], collect_state[2],
                    ref_x, ref_y, ref_th
                ])
                Y_data.append([ctrl[0], ctrl[1]])

            # step unicycle
            collect_state = dynamics.step(collect_state, ctrl)
            local_traj.append(collect_state.copy())

            iteration_count += 1
            step_counter    += 1

        mpc_trajectories[name] = np.array(local_traj)
        print(f"  -> {name}: coverage={coverage:.3f}, steps={iteration_count}, data so far={len(X_data)}.")

        if len(X_data) >= replay_buffer_size:
            print("Replay buffer limit => stop collecting.")
            break

    return np.array(X_data), np.array(Y_data), mpc_trajectories
