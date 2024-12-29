"""
mpc_net/paths/path_generators.py

Defines various parameterized path generator functions.
We now systematically add multiple variants for each path:
  - multiple figure-8 scales
  - multiple circle radii
  - multiple spirals, etc.

Each function returns shape (2, T) => [x array; y array].
"""

import numpy as np

def long_line(T=1500, length=20):
    x = np.linspace(0, length, T)
    y = np.zeros_like(x)
    return np.vstack((x, y))

def bigger_circle(T=1500, radius=8):
    """
    Circle with radius=8 by default. We'll param further below.
    """
    t = np.linspace(0, 2*np.pi, T)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.vstack((x, y))

def vertical_fig8(T=1500, a=8, b=3):
    """
    Figure-8 oriented more vertically:
      x = a*sin(t), y = b*sin(t)*cos(t)
    We'll param further below with multiple (a,b).
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
    """
    r = a + b*t
    x = r*cos(t), y=r*sin(t)
    We'll param with multiple (a,b) below.
    """
    t = np.linspace(0, 4*np.pi, T)
    r = a + b*t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.vstack((x, y))

def elliptical_path(T=1500, a=12, b=6):
    """
    Ellipse: x=a*cos(t), y=b*sin(t)
    """
    t = np.linspace(0, 2*np.pi, T)
    x = a*np.cos(t)
    y = b*np.sin(t)
    return np.vstack((x, y))

def generate_paths():
    """
    Return a list of (name, function) pairs, systematically
    enumerating multiple variants of each path type.
    """
    paths = []

    # 1) Lines with multiple lengths
    for L in [10, 20, 30]:
        def make_line(ll=L):
            return long_line(T=1500, length=ll)
        paths.append((f"Line length={L}", make_line))

    # 2) Circles with multiple radii
    for r in [5, 8, 12]:
        def make_circle(rr=r):
            return bigger_circle(T=1500, radius=rr)
        paths.append((f"Circle radius={r}", make_circle))

    # 3) Figure-8 with multiple (a,b) combos
    #    e.g. (8,3), (10,4), (12,6)
    for (aa, bb) in [(8,3), (10,4), (12,6)]:
        def make_fig8(A=aa, B=bb):
            return vertical_fig8(T=1500, a=A, b=B)
        paths.append((f"Fig8 a={aa},b={bb}", make_fig8))

    # 4) Sinusoids with multiple amplitudes/frequencies
    #    e.g. amplitude in [2,4], frequency in [1,2]
    for amp in [2,4]:
        for freq in [1,2]:
            def make_sin(A=amp, F=freq):
                return mild_sinusoid(T=1500, amplitude=A, frequency=F)
            paths.append((f"Sinusoid amp={amp}, freq={freq}", make_sin))

    # 5) Spirals with multiple (a,b)
    #    e.g. (2,0.2), (3,0.3), (4,0.5)
    for (aa, bb) in [(2,0.2), (3,0.3), (4,0.5)]:
        def make_spiral(A=aa, B=bb):
            return mild_spiral(T=1500, a=A, b=B)
        paths.append((f"Spiral a={aa}, b={bb}", make_spiral))

    # 6) Ellipses
    for (ea, eb) in [(10,5), (12,6), (14,4)]:
        def make_ellipse(EA=ea, EB=eb):
            return elliptical_path(T=1500, a=EA, b=EB)
        paths.append((f"Ellipse a={ea},b={eb}", make_ellipse))

    return paths
