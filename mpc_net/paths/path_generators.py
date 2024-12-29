"""
mpc_net/paths/path_generators.py

Simpler, less-dense paths (T=1500 or 2000), so fewer points => 
faster coverage checks and fewer solver calls.
"""

import numpy as np

def long_line(T=1500, length=20):
    x = np.linspace(0, length, T)
    y = np.zeros_like(x)
    return np.vstack((x, y))

def bigger_circle(T=1500, radius=8):
    t = np.linspace(0, 2*np.pi, T)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.vstack((x, y))

def vertical_fig8(T=1500, a=8, b=3):
    """
    More vertical figure-8:
      x = a*sin(t), y = b*sin(t)*cos(t).
    Using T=1500 => fewer points => less overhead.
    """
    t = np.linspace(0, 2*np.pi, T)
    x = a * np.sin(t)
    y = b * np.sin(t) * np.cos(t)
    return np.vstack((x, y))

def mild_sinusoid(T=1500, amplitude=4, frequency=1):
    x = np.linspace(0, 15, T)
    y = amplitude * np.sin(frequency * x * 0.5)
    return np.vstack((x, y))

def mild_spiral(T=1500, a=2.0, b=0.2):
    t = np.linspace(0, 4*np.pi, T)
    r = a + b*t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.vstack((x, y))

def generate_paths():
    """
    Return a list of (name, function) pairs. 
    Keep them simpler/less-dense => T=1500 => fewer path points.
    """
    paths = []

    # 1) Long line
    paths.append(("Line length=20", lambda: long_line(T=1500, length=20)))

    # 2) Larger circles
    for r in [6, 8]:
        paths.append((f"Circle radius={r}",
                      lambda rr=r: bigger_circle(T=1500, radius=rr)))

    # 3) Vertical fig8
    paths.append(("Fig8 a=8,b=3",
                  lambda: vertical_fig8(T=1500, a=8, b=3)))

    # 4) Mild sinusoid
    paths.append(("Sinusoid amp=4,freq=1",
                  lambda: mild_sinusoid(T=1500, amplitude=4, frequency=1)))

    # 5) Mild spiral
    paths.append(("Spiral a=2.0,b=0.2",
                  lambda: mild_spiral(T=1500, a=2.0, b=0.2)))

    return paths
