import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Golden ratio
# -----------------------------
phi = (1 + np.sqrt(5)) / 2
pi = np.pi

# -----------------------------
# Affine transform utilities
# -----------------------------
def scale(s):
    M = np.array([[s, 0], [0, s]])
    b = np.array([0.0, 0.0])
    return M, b

def shift(v):
    M = np.eye(2)
    b = np.array(v, dtype=float)
    return M, b

def rotate(theta):
    c, s = np.cos(theta), np.sin(theta)
    M = np.array([[c, -s], [s, c]])
    b = np.array([0.0, 0.0])
    return M, b

def compose(T2, T1):
    # T2(T1(x)) = M2*(M1*x + b1) + b2 = (M2*M1)x + (M2*b1 + b2)
    M1, b1 = T1
    M2, b2 = T2
    return M2 @ M1, M2 @ b1 + b2

# -----------------------------
# Build the transforms (SS edge omitted)
# -----------------------------

# RR: rectangle → rectangle
RR = compose(
        compose(
            shift([phi, 0]),
            rotate(pi/2)
        ),
        scale(1/phi)
     )

# RS: rectangle → square (kept as in original)
RS = scale(1)

# SR1: square → rectangle (lower left)
SR1 = scale(1 / phi**2)

# SR2: square → rectangle (lower right)
SR2 = compose(
        compose(
            shift([1/phi + 1/phi**2, 0]),
            scale(1 / phi**2)
        ),
        rotate(pi/2)
     )

# SR3: square → rectangle (upper left)
SR3 = compose(
        compose(
            shift([1/phi**2, 1/phi**2]),
            scale(1 / phi**2)
        ),
        rotate(pi/2)
     )

# SR4: square → rectangle (upper right)
SR4 = compose(
        shift([1/phi**2, 1/phi]),
        scale(1 / phi**2)
     )

# -----------------------------
# Digraph edges (SS edge removed)
# -----------------------------
edges = [
    ("R", "R", RR),
    ("R", "S", RS),
    ("S", "R", SR1),
    ("S", "R", SR2),
    ("S", "R", SR3),
    ("S", "R", SR4)
]

# Group edges by source node
edges_from = {"R": [], "S": []}
for src, tgt, T in edges:
    edges_from[src].append((tgt, T))

# -----------------------------
# Run the digraph IFS
# -----------------------------
def run_digraph_ifs(n_points=2000000):
    pts = np.zeros((n_points, 2))
    colors = np.zeros(n_points)

    node = "R"
    p = np.array([0.0, 0.0])

    for i in range(n_points):
        tgt, T = edges_from[node][np.random.randint(len(edges_from[node]))]
        M, b = T

        p = M @ p + b

        pts[i] = p
        colors[i] = 0 if node == "R" else 1

        node = tgt

    return pts, colors

pts, colors = run_digraph_ifs(2000000)

# -----------------------------
# Plot and export (different filename)
# -----------------------------
plt.figure(figsize=(8, 8))
plt.scatter(pts[:,0], pts[:,1], c=colors, cmap="turbo", s=0.1)
plt.axis("equal")
plt.axis("off")
plt.savefig("mccclure_digraph_ifs_noSS_2000k.png", dpi=2000, bbox_inches="tight")
plt.close()
