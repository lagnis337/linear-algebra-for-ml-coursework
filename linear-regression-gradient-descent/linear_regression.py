"""Linear regression two ways: the normal equation and batch gradient descent.

Python port of the Octave scripts submitted for UCSD Extended Studies
CSE X412.87 (Week 8). Reproduces the submitted results exactly and regenerates
the figures, including the two bonus ones (GD trajectory on the cost surface,
learning-rate comparison).

    python linear_regression.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
X2 = np.loadtxt(HERE / "ex3x.txt")   # living area (sq ft), bedrooms
y = np.loadtxt(HERE / "ex3y.txt")    # price (USD)
x = X2[:, 0]                          # living area only, per the brief
m = len(y)
QUERY_SQFT = 1650


def design(x_col):
    """Prepend the bias column of ones."""
    return np.c_[np.ones(len(x_col)), x_col]


def mse(A, w, y):
    r = y - A @ w
    return (r @ r) / m


# ---------------------------------------------------- 1. normal equation
# w = (XᵀX)⁻¹ Xᵀy, on the raw (unnormalised) feature.
A = design(x)
w_neq = np.linalg.solve(A.T @ A, A.T @ y)
price_neq = np.array([1, QUERY_SQFT]) @ w_neq

# --------------------------------------------------- 2. gradient descent
# Standardise the feature first so the cost surface is well conditioned.
# Octave's std() divides by n-1, so use ddof=1 to match the submitted numbers.
mu, sigma = x.mean(), x.std(ddof=1)
An = design((x - mu) / sigma)

ALPHA, ITERS = 0.07, 100


def gradient_descent(A, y, alpha, iters):
    w = np.zeros(2)
    path, cost = [], []
    for _ in range(iters):
        h = A @ w
        cost.append(((y - h) @ (y - h)) / m)
        grad = -2 / m * A.T @ (y - h)         # ∇J = -(2/m) Xᵀ(y - Xw)
        w = w - alpha * grad
        path.append(w.copy())
    return w, np.array(cost), np.array(path)


w_gd, cost, path = gradient_descent(An, y, ALPHA, ITERS)
price_gd = np.array([1, (QUERY_SQFT - mu) / sigma]) @ w_gd

print(f"{'':16}{'Normal eq':>14}{'Gradient descent':>18}")
print(f"{'Price of 1650sf':16}{price_neq:>14,.2f}{price_gd:>18,.2f}")
print(f"{'w0':16}{w_neq[0]:>14,.2f}{w_gd[0]:>18,.2f}")
print(f"{'w1':16}{w_neq[1]:>14,.2f}{w_gd[1]:>18,.2f}")
print(f"\nGD cost: {cost[0]:.3e} at iteration 1 -> {cost[-1]:.3e} at iteration {ITERS}")

# ------------------------------------------------------------ figures
# 1. Cost surface over a (w0, w1) grid, with the GD path overlaid (bonus).
w0s = np.linspace(100_000, 500_000, 100)
w1s = np.linspace(1_000, 300_000, 100)
W0, W1 = np.meshgrid(w0s, w1s)
J = np.array([[mse(An, np.array([a, b]), y) for a in w0s] for b in w1s])

fig = plt.figure(figsize=(7, 5.5))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(W0, W1, J, cmap="viridis", alpha=0.75, linewidth=0)
ax.plot(path[:, 0], path[:, 1], cost, "rv-", markersize=4, label="GD iterates")
ax.set_xlabel("w0"); ax.set_ylabel("w1"); ax.set_zlabel("MSE")
ax.set_title("Cost surface with gradient-descent trajectory")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "cost_surface_trajectory.png", dpi=120)

# 2. Cost vs iteration.
plt.figure(figsize=(6, 4))
plt.plot(range(ITERS), cost)
plt.xlabel("Iteration"); plt.ylabel("Cost (MSE)")
plt.title(f"Gradient descent, alpha = {ALPHA}")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIG / "cost_vs_iterations.png", dpi=120)

# 3. Fitted line over the data (both methods coincide).
plt.figure(figsize=(6, 4))
plt.plot(x, y, "o", alpha=0.7, label="training data")
xs = np.linspace(x.min(), x.max(), 50)
plt.plot(xs, design(xs) @ w_neq, "-", label="normal equation")
plt.plot(xs, design((xs - mu) / sigma) @ w_gd, "--", label="gradient descent")
plt.xlabel("Living area (sq ft)"); plt.ylabel("Price (USD)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIG / "regression_line.png", dpi=120)

# 4. Learning-rate comparison (bonus): too small converges slowly, too large diverges.
plt.figure(figsize=(6, 4))
for alpha in [0.01, 0.03, 0.07, 0.3, 1.0, 1.05]:
    _, c, _ = gradient_descent(An, y, alpha, ITERS)
    plt.plot(range(ITERS), c, label=f"alpha = {alpha}")
plt.yscale("log")
plt.xlabel("Iteration"); plt.ylabel("Cost (MSE, log scale)")
plt.title("Effect of the learning rate")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
plt.savefig(FIG / "learning_rate_comparison.png", dpi=120)
print(f"\nFigures written to {FIG.relative_to(HERE.parent)}/")
