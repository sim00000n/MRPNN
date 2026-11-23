#!/usr/bin/env python3
"""
Evenly distribute 8 samples inside an 8×8 square, with a user-specified minimum
distance to edges (an inset "free" quad). No tiling / neighbor tiles.

Method: discrete Lloyd relaxation (centroidal Voronoi tessellation) on a grid.
- Create a uniform grid of points inside the inset rectangle [m, 8-m]²
- Assign each grid point to its nearest sample (Voronoi regions, discretized)
- Move each sample to the centroid of its assigned region
- Repeat for FRAMES iterations

This tends to equalize the *area* owned by each sample and produces a very even spread.
Only uses numpy + matplotlib.

Requirements for Qt window:
  pip install PyQt6  (or PySide6 / PyQt5)
"""

import matplotlib
matplotlib.use("QtAgg")  # must be before pyplot

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# User parameters
# ----------------------------
L = 8.0
N = 8

MIN_EDGE_DIST = 0.0     # <-- your inset margin (samples must stay within [m, L-m])
FRAMES = 200            # iterations
GRID_RES = 220          # resolution per axis for Lloyd assignment (higher = smoother, slower)

RELAX = 0.65            # 0..1, how strongly to move toward centroid each iteration
JITTER = 0.2            # initial jitter
SEED = 2

# Optional: keep samples from collapsing early (tiny repulsion inside inset)
USE_SOFT_REPULSION = True
REPULSE_K = 0.02        # small
REPULSE_CUTOFF = 0.55   # in "target spacing" units (see d_target below)


# ----------------------------
# Derived values
# ----------------------------
m = float(MIN_EDGE_DIST)
if not (0.0 <= m < L / 2):
    raise ValueError("MIN_EDGE_DIST must be in [0, L/2).")

xmin, xmax = m, L - m
ymin, ymax = m, L - m
W = xmax - xmin
H = ymax - ymin
AREA = W * H

d_target = np.sqrt(AREA / N)  # rough spacing scale inside inset


def clamp_inset(P: np.ndarray) -> np.ndarray:
    P[:, 0] = np.clip(P[:, 0], xmin, xmax)
    P[:, 1] = np.clip(P[:, 1], ymin, ymax)
    return P


def make_grid(res: int):
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    G = np.stack([X.ravel(), Y.ravel()], axis=1)  # (res*res, 2)
    return G, xs, ys


def lloyd_step(P: np.ndarray, G: np.ndarray, relax: float):
    """
    One discrete Lloyd iteration:
    - Assign grid points to nearest sample
    - Move each sample toward centroid of its assigned points
    """
    # Distances: (M, N)
    # M = number of grid points
    diff = G[:, None, :] - P[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    owner = np.argmin(dist2, axis=1)

    newP = P.copy()
    for i in range(N):
        pts = G[owner == i]
        if len(pts) == 0:
            # rare: if a site loses all area, jitter it back in
            newP[i] = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
            ])
        else:
            c = pts.mean(axis=0)
            newP[i] = (1.0 - relax) * P[i] + relax * c

    return clamp_inset(newP), owner


def soft_repulsion(P: np.ndarray):
    """Tiny repulsion to discourage near-collisions (optional)."""
    F = np.zeros_like(P)
    r = REPULSE_CUTOFF * d_target
    r2 = r * r
    eps = 1e-6
    for i in range(N):
        for j in range(i + 1, N):
            d = P[i] - P[j]
            d2 = float(d @ d) + eps
            if d2 < r2:
                dist = np.sqrt(d2)
                dir_ = d / dist
                # push apart proportional to (r - dist)
                mag = REPULSE_K * (r - dist) / (dist + eps)
                F[i] += mag * dir_
                F[j] -= mag * dir_
    return F


# ----------------------------
# Init samples inside inset
# ----------------------------
rng = np.random.default_rng(SEED)
P = np.column_stack([
    rng.uniform(xmin, xmax, size=N),
    rng.uniform(ymin, ymax, size=N),
])
P += rng.normal(scale=JITTER, size=P.shape)
P = clamp_inset(P)

G, xs, ys = make_grid(GRID_RES)

# ----------------------------
# Visualization setup
# ----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_title("Lloyd relaxation inside inset quad")

# Outer boundary
ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0])

# Inset boundary
ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])

# Scatter plot for grid assignment (shown as faint colored pixels)
# We'll update it via an image for speed.
img = ax.imshow(
    np.zeros((GRID_RES, GRID_RES)),
    extent=(xmin, xmax, ymin, ymax),
    origin="lower",
    interpolation="nearest",
    alpha=0.35,
)

sc = ax.scatter(P[:, 0], P[:, 1], s=60)
labels = [ax.text(P[i, 0], P[i, 1], str(i), fontsize=9, ha="left", va="bottom") for i in range(N)]
info = ax.text(0.02, 1.02, "", transform=ax.transAxes)


def update_plot(owner):
    # owner is (GRID_RES*GRID_RES,)
    owner_img = owner.reshape((GRID_RES, GRID_RES))
    img.set_data(owner_img)

    sc.set_offsets(P)
    for i in range(N):
        labels[i].set_position((P[i, 0], P[i, 1]))

    # simple quality metric: nearest-neighbor distances
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2) + np.eye(N) * 1e9)
    nn = D.min(axis=1)
    info.set_text(
        f"margin={m:.2f}  target d≈{d_target:.3f}  nn min/mean={nn.min():.3f}/{nn.mean():.3f}"
    )

    fig.canvas.draw_idle()
    plt.pause(0.01)


# ----------------------------
# Main loop
# ----------------------------
for frame in range(FRAMES):
    P, owner = lloyd_step(P, G, RELAX)

    if USE_SOFT_REPULSION:
        P = clamp_inset(P + soft_repulsion(P))

    P[0] = (1.5, 1)
    P[1] = (1.5, 7)
    P[2] = (6.5, 1)
    P[3] = (6.5, 7)
    P[4] = (4, 2.5)
    P[5] = (4, 5.5)
    P[6] = (1, 4)
    P[7] = (7, 4)

    if frame % 1 == 0:
        update_plot(owner)

plt.ioff()
plt.show()

print("Final sample positions:")
for i, (x, y) in enumerate(P):
    print(f"p{i}: ({x - 4.0:.6f}, {y - 4.0:.6f})")
