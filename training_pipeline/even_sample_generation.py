#!/usr/bin/env python3
import matplotlib
matplotlib.use("QtAgg")  # must be before pyplot

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
L = 8.0
N = 8
FRAMES = 1400

DT = 0.055
DAMPING = 0.92

# Springs (local structure)
K_NEIGHBORS = 5
SPRING_K = 0.5

# Repulsion (prevents clumping; periodic)
REPULSE_K = 0.70
REPULSE_CUTOFF = 2      # repel if dist < cutoff*d_point

# NEW: gentle edge avoidance (only near walls)
EDGE_AVOID_K = 5        # small on purpose
EDGE_MARGIN = 1         # start pushing inward when closer than this to an edge

# Init / stability
MARGIN = 0.9
JITTER = 0.5
MAX_FORCE = 4.5

# Draw
DRAW_EDGES = True
EDGE_ALPHA = 0.20
TILE_PREVIEW_N = 5

# Target spacing estimate
d_point = np.sqrt((L * L) / N)
r_repulse = REPULSE_CUTOFF * d_point

# 9-tile offsets (center + 8 neighbors)
OFFSETS = np.array([(dx, dy) for dx in (-L, 0.0, L) for dy in (-L, 0.0, L)], dtype=float)


# ----------------------------
# Helpers
# ----------------------------
def clamp(P: np.ndarray):
    return np.clip(P, 0.0, L)

def build_P2(P: np.ndarray) -> np.ndarray:
    return np.vstack([P + off for off in OFFSETS])

def best_image_dvec(Pi: np.ndarray, Pj: np.ndarray):
    candidates = Pj + OFFSETS
    dvecs = candidates - Pi
    dists = np.linalg.norm(dvecs, axis=1)
    m = int(np.argmin(dists))
    return dvecs[m], float(dists[m]), candidates[m]

def knn_edges_periodic(P: np.ndarray, k: int):
    best_dvec = {}
    best_dist = np.full((N, N), np.inf, dtype=float)

    for i in range(N):
        for j in range(i + 1, N):
            dvec, dist, _ = best_image_dvec(P[i], P[j])
            best_dvec[(i, j)] = dvec
            best_dist[i, j] = dist
            best_dist[j, i] = dist

    edges = set()
    for i in range(N):
        order = np.argsort(best_dist[i])
        nn = [j for j in order if j != i][:k]
        for j in nn:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    return sorted(edges), best_dvec


# ----------------------------
# Simulation step
# ----------------------------
def step(P: np.ndarray, V: np.ndarray, P2: np.ndarray):
    F = np.zeros_like(P)

    # --- periodic short-range repulsion (all pairs) ---
    for i in range(N):
        for j in range(i + 1, N):
            dvec, dist, _ = best_image_dvec(P[i], P[j])
            if dist < 1e-8:
                continue
            if dist < r_repulse:
                dir_ = dvec / dist
                mag = REPULSE_K * (r_repulse - dist) / dist
                fij = mag * dir_
                F[i] -= fij
                F[j] += fij

    # --- periodic kNN springs toward d_point ---
    minL = 1000
    maxL = 0
    edges, best_dvec = knn_edges_periodic(P, K_NEIGHBORS)
    for i, j in edges:
        dvec = best_dvec[(i, j)]
        dist = np.linalg.norm(dvec)
        if dist < 1e-8:
            continue
        minL = min(dist, minL)
        maxL = max(dist, maxL)
        dir_ = dvec / dist
        mag = SPRING_K * (dist - d_point)
        fij = mag * dir_
        F[i] += fij
        F[j] -= fij

    print("Min: %.3f, Max: %.3f" % (minL, maxL))

    # --- NEW: gentle edge-avoidance barrier (only near walls) ---
    # Push inward if distance to a wall < EDGE_MARGIN
    x = P[:, 0]
    y = P[:, 1]

    # left wall: dist = x
    left = x
    mask = left < EDGE_MARGIN
    if np.any(mask):
        # push +x
        F[mask, 0] += EDGE_AVOID_K * (EDGE_MARGIN - left[mask]) / (left[mask] + 1e-6)

    # right wall: dist = L - x
    right = L - x
    mask = right < EDGE_MARGIN
    if np.any(mask):
        # push -x
        F[mask, 0] -= EDGE_AVOID_K * (EDGE_MARGIN - right[mask]) / (right[mask] + 1e-6)

    # bottom wall: dist = y
    bottom = y
    mask = bottom < EDGE_MARGIN
    if np.any(mask):
        # push +y
        F[mask, 1] += EDGE_AVOID_K * (EDGE_MARGIN - bottom[mask]) / (bottom[mask] + 1e-6)

    # top wall: dist = L - y
    top = L - y
    mask = top < EDGE_MARGIN
    if np.any(mask):
        # push -y
        F[mask, 1] -= EDGE_AVOID_K * (EDGE_MARGIN - top[mask]) / (top[mask] + 1e-6)

    # Clip for stability
    norms = np.linalg.norm(F, axis=1)
    too_big = norms > MAX_FORCE
    if np.any(too_big):
        F[too_big] *= (MAX_FORCE / norms[too_big])[:, None]

    V = DAMPING * (V + DT * F)
    P = clamp(P + DT * V)
    return P, V, edges


# ----------------------------
# Init
# ----------------------------
rng = np.random.default_rng(2)
P = rng.uniform(MARGIN, L - MARGIN, size=(N, 2))
P += rng.normal(scale=JITTER, size=P.shape)
P = clamp(P)
V = np.zeros_like(P)

# ----------------------------
# Visualization (base + tiling preview)
# ----------------------------
plt.ion()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Periodic tiling relaxation (edge-avoidance added)", y=0.98)

# Base cell
ax0.set_aspect("equal", adjustable="box")
ax0.set_xlim(0, L)
ax0.set_ylim(0, L)
ax0.set_title("")
ax0.plot([0, L, L, 0, 0], [0, 0, L, L, 0])

sc0 = ax0.scatter(P[:, 0], P[:, 1])
texts0 = [ax0.text(P[i, 0], P[i, 1], str(i), fontsize=9, ha="left", va="bottom") for i in range(N)]
info0 = ax0.text(0.02, 1.02, "", transform=ax0.transAxes)
edge_lines = []

# Tiled preview
T = TILE_PREVIEW_N
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlim(0, L * T)
ax1.set_ylim(0, L * T)
ax1.set_title(f"{T}×{T} tiling preview")

for k in range(T + 1):
    ax1.plot([0, L * T], [k * L, k * L], linewidth=0.6, alpha=0.5)
    ax1.plot([k * L, k * L], [0, L * T], linewidth=0.6, alpha=0.5)

tiles_offsets = np.array([(tx * L, ty * L) for tx in range(T) for ty in range(T)], dtype=float)
P_tiled = np.vstack([P + off for off in tiles_offsets])
sc1 = ax1.scatter(P_tiled[:, 0], P_tiled[:, 1])

# ----------------------------
# Main loop
# ----------------------------
for frame in range(FRAMES):
    P2 = build_P2(P)
    P, V, edges = step(P, V, P2)

    # update base
    sc0.set_offsets(P)
    for i in range(N):
        texts0[i].set_position((P[i, 0], P[i, 1]))

    if DRAW_EDGES:
        for ln in edge_lines:
            ln.remove()
        edge_lines = []
        for i, j in edges:
            _, _, pj_img = best_image_dvec(P[i], P[j])
            ln, = ax0.plot([P[i, 0], pj_img[0]], [P[i, 1], pj_img[1]], alpha=EDGE_ALPHA)
            edge_lines.append(ln)

    # update tiled preview
    P_tiled = np.vstack([P + off for off in tiles_offsets])
    sc1.set_offsets(P_tiled)

    info0.set_text(
        f"frame={frame}/{FRAMES-1}  d≈{d_point:.3f}  repel<{r_repulse:.3f}  edge_margin={EDGE_MARGIN:.2f}"
    )

    fig.canvas.draw_idle()
    plt.pause(0.02)

    if(frame % 50 == 49):
        print("Frame %i sample positions (base cell):" % frame)
        for i, (px, py) in enumerate(P):
            print(f"p{i}: ({px:.6f}, {py:.6f})")

plt.ioff()
plt.show()

print("Final sample positions (base cell):")
for i, (px, py) in enumerate(P):
    print(f"p{i}: ({px:.6f}, {py:.6f})")
