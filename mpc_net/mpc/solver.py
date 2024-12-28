# mpc_net/mpc/solver.py

import numpy as np
from casadi import Opti, cos, sin, vertcat

def solve_mpc(initial_state, ref_path, constraints,
              dt=0.01, horizon_steps=50, max_iter=10000,
              v0=0.0, omega0=0.0):
    """
    Solve single-step MPC, return U_sol shape (2, N) or empty if failure.
    ref_path: shape (3, N_ref) => [x, y, yaw_ref].
    We'll only apply the first control from U_sol.
    """
    N = min(horizon_steps, ref_path.shape[1])
    if N < 1:
        return np.array([]), np.array([])
    
    opti = Opti()
    X = opti.variable(3, N+1)
    U = opti.variable(2, N)

    # Example weights
    weights = {
        'v_smooth':     500.0,
        'w_smooth':     500.0,
        'cte':        40000.0,
        'te':           300.0,
        'lateral_acc': 100.0
    }

    J = 0
    for t in range(N):
        v = U[0,t]
        omega = U[1,t]
        dv = U[0,t] - (U[0,t-1] if t>0 else v0)
        dw = U[1,t] - (U[1,t-1] if t>0 else omega0)

        cte = (X[0,t] - ref_path[0,t])**2 + (X[1,t] - ref_path[1,t])**2
        te  = (X[2,t] - ref_path[2,t])**2
        lat_acc = v*omega

        J += (weights['v_smooth']*dv**2 +
              weights['w_smooth']*dw**2 +
              weights['cte']*cte +
              weights['te']*te +
              weights['lateral_acc']*lat_acc**2)

    opti.minimize(J)

    # dynamics
    for t in range(N):
        x_next = X[0,t] + U[0,t]*cos(X[2,t])*dt
        y_next = X[1,t] + U[0,t]*sin(X[2,t])*dt
        th_next= X[2,t] + U[1,t]*dt
        opti.subject_to(X[:,t+1] == vertcat(x_next, y_next, th_next))

        opti.subject_to(constraints['v_min'] <= U[0,t])
        opti.subject_to(U[0,t] <= constraints['v_max'])
        opti.subject_to(constraints['omega_min'] <= U[1,t])
        opti.subject_to(U[1,t] <= constraints['omega_max'])

    # initial condition
    opti.subject_to(X[:,0] == initial_state)

    # solver
    opti.solver("ipopt", {"ipopt": {"max_iter": max_iter}})
    try:
        sol = opti.solve()
        U_sol = sol.value(U)
        X_sol = sol.value(X)
        # shape => U_sol is (2, N)
        return U_sol, X_sol
    except RuntimeError:
        print("MPC solver failed => returning empty.")
        return np.array([]), np.array([])
