# ============================================================
#  Kleinian Group Explorer (Generalized for N Generators)
#  Supports your 4-generator torus example
# ============================================================

import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import json
import math

# ------------------------------------------------------------
# 1. Normalize PSL(2,C) matrices
# ------------------------------------------------------------

def normalize_matrix(M):
    M = np.array(M, dtype=complex)
    d = np.linalg.det(M)
    if d == 0:
        raise ValueError("Matrix determinant is zero")
    return M / (d ** 0.5)

# ------------------------------------------------------------
# 2. Your 4-generator torus group
# ------------------------------------------------------------

w = np.exp(1j * 2 * np.pi / 3)

s1 = normalize_matrix([[w, 0], [-2, np.conj(w)]])
s2 = normalize_matrix([[1j*w, -2], [0, -1j*np.conj(w)]])
s3 = normalize_matrix([[0, w], [-np.conj(w), 2]])
s4 = normalize_matrix([[0, 1j*w], [1j*np.conj(w), 2]])

generators = {
    's1': s1,
    's2': s2,
    's3': s3,
    's4': s4
}

# ------------------------------------------------------------
# 3. Möbius action
# ------------------------------------------------------------

def mobius_apply(M, z):
    a, b = M[0,0], M[0,1]
    c, d = M[1,0], M[1,1]
    if z is None:
        return a/c if abs(c) > 0 else None
    denom = c*z + d
    if abs(denom) == 0:
        return None
    return (a*z + b) / denom

# ------------------------------------------------------------
# 4. Isometric hemispheres
# ------------------------------------------------------------

p0 = (0.0, 0.0, 1.0)

def isometric_hemisphere(M, base=p0):
    x0, y0, t0 = base
    z0 = x0 + 1j*y0
    w = mobius_apply(M, z0)
    if w is None:
        return None
    c = 0.5*(z0 + w)
    r = math.sqrt(abs(z0 - c)**2 + t0**2)
    return (c.real, c.imag, r)

def sample_hemisphere(cx, cy, r, n_theta=40, n_phi=20):
    thetas = np.linspace(0, 2*np.pi, n_theta)
    phis = np.linspace(0, np.pi/2, n_phi)
    X, Y, T = [], [], []
    for phi in phis:
        for theta in thetas:
            X.append(cx + r*np.cos(theta)*np.sin(phi))
            Y.append(cy + r*np.sin(theta)*np.sin(phi))
            T.append(r*np.cos(phi))
    return np.array(X), np.array(Y), np.array(T)

# ------------------------------------------------------------
# 5. Limit set approximation
# ------------------------------------------------------------

def approximate_limit_set(gens, n_points=3000, word_length=12):
    keys = list(gens.keys())
    pts = []
    for _ in range(n_points):
        z = np.exp(1j * 2*np.pi * random.random())
        w = z
        for _ in range(word_length):
            g = gens[random.choice(keys)]
            w = mobius_apply(g, w)
            if w is None:
                break
        if w is not None and abs(w) < 1e6:
            pts.append(w)
    return np.array([p.real for p in pts]), np.array([p.imag for p in pts])

# ------------------------------------------------------------
# 6. Plot hemispheres + limit set
# ------------------------------------------------------------

def plot_group(gens):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red','green','blue','purple','orange','cyan']
    hemisphere_data = {}

    for idx, (name, M) in enumerate(gens.items()):
        hemi = isometric_hemisphere(M)
        if hemi is None:
            continue
        cx, cy, r = hemi
        hemisphere_data[name] = {'center':[cx,cy], 'radius':r}
        X,Y,T = sample_hemisphere(cx,cy,r)
        ax.plot_trisurf(X,Y,T, color=colors[idx%len(colors)], alpha=0.4)
        ax.text(cx,cy,r,name)

    Xlim, Ylim = approximate_limit_set(gens)
    ax.scatter(Xlim, Ylim, np.zeros_like(Xlim), s=1, color='black')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title("4-Generator Kleinian Group (Torus Example)")
    plt.show()

    return hemisphere_data

# ------------------------------------------------------------
# 7. Run
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Generators:")
    for k,v in generators.items():
        print(k, "=", v)
    hemi = plot_group(generators)
    with open("kleinian_4group_data.json","w") as f:
        json.dump