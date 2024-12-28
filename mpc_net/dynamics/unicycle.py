# mpc_net/dynamics/unicycle.py

import numpy as np

class UnicycleDynamics:
    def __init__(self, dt=0.01):
        self.dt = dt

    def step(self, state, control):
        """
        state = [x, y, theta]
        control = [v, omega]
        Returns next state after one step.
        """
        x, y, theta = state
        v, omega = control
        nx = x + v * np.cos(theta) * self.dt
        ny = y + v * np.sin(theta) * self.dt
        nth = theta + omega * self.dt
        return np.array([nx, ny, nth])