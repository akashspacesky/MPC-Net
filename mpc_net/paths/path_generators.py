# mpc_net/paths/path_generators.py

import numpy as np

def long_line(T=2500, length=20):
    x = np.linspace(0, length, T)
    y = np.zeros_like(x)
    return np.vstack((x, y))

def bigger_circle(T=2500, radius=8):
    t = np.linspace(0, 2*np.pi, T)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.vstack((x, y))

def vertical_fig8(T=2500, a=8, b=3):
    """
    A r, more vertical figure-8:
      x = b*sin(t), y = a*sin(t)*cos(t)
    """
    t = np.linspace(0, 2*np.pi, T)
    x = a * np.sin(t)
    y = b * np.sin(t) * np.cos(t)
    return np.vstack((x, y))

def mild_sinusoid(T=2500, amplitude=4, frequency=1):
    x = np.linspace(0, 15, T)
    y = amplitude * np.sin(frequency * x * 0.5)
    return np.vstack((x, y))

def mild_spiral(T=2500, a=2.0, b=0.2):
    t = np.linspace(0, 4*np.pi, T)
    r = a + b*t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.vstack((x, y))

def generate__paths():
    """
    Return list of (name, function) pairs for bigger/less-sharp paths.
    """
    paths = []

    # Long line
    paths.append(("Line length=20", lambda: long_line(T=2500, length=20)))
    # Circles
    for r in [6, 8, 10]:
        paths.append((f"Circle radius={r}", lambda rr=r: bigger_circle(T=2500, radius=rr)))
    #  figure-8
    for (aa, bb) in [(8, 6), (8, 6)]:
        paths.append((f"Fig8 a={aa},b={bb}", lambda A=aa,B=bb: vertical_fig8(T=4000,a=A,b=B)))
    # Mild sinusoid
    paths.append(("Sinusoid amp=4,freq=1", lambda: mild_sinusoid(T=2500, amplitude=4, frequency=1)))
    # Mild spiral
    paths.append(("Spiral a=2.0,b=0.2", lambda: mild_spiral(T=2500,a=5.0,b=2.0)))

    return paths
