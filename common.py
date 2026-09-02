import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from pathlib import Path

TOL = 1e-6
MAX_ITER = 2000

INITIAL_POINTS = [
    np.array([-2.0, 2.0]),
    np.array([0.5, -1.5]),
    np.array([3.0, 3.0]),
]

LEARNING_RATES = [0.001, 0.01, 0.1]

def f1(x):
    a, b = x
    return a*a + b*b

def grad_f1(x):
    a, b = x
    return np.array([2*a, 2*b], dtype=float)

def hess_f1(x):
    return np.array([[2.0, 0.0], [0.0, 2.0]])

def f2(x):
    a, b = x
    return (1-a)**2 + 100*(b-a*a)**2

def grad_f2(x):
    a, b = x
    return np.array([
        -2*(1-a) - 400*a*(b-a*a),
        200*(b-a*a)
    ], dtype=float)

def hess_f2(x):
    a, b = x
    return np.array([
        [2 - 400*b + 1200*a*a, -400*a],
        [-400*a, 200]
    ], dtype=float)

def f3(x):
    a, b = x
    return a*a + b*b + 10*np.cos(a) + 10*np.cos(b)

def grad_f3(x):
    a, b = x
    return np.array([
        2*a - 10*np.sin(a),
        2*b - 10*np.sin(b)
    ], dtype=float)

def hess_f3(x):
    a, b = x
    return np.array([
        [2 - 10*np.cos(a), 0.0],
        [0.0, 2 - 10*np.cos(b)]
    ], dtype=float)

FUNCTIONS = {
    "quadratic": (f1, grad_f1, hess_f1, (-4, 4), (-4, 4)),
    "rosenbrock": (f2, grad_f2, hess_f2, (-3, 3), (-2, 4)),
    "cosine_bumps": (f3, grad_f3, hess_f3, (-8, 8), (-8, 8)),
}

def ensure_output_dir(method):
    out = Path("output") / method
    out.mkdir(parents=True, exist_ok=True)
    return out

def save_results(method, rows):
    out = ensure_output_dir(method)
    path = out / "results.csv"
    fields = [
        "function", "start_x", "start_y", "learning_rate",
        "iterations", "converged", "final_x", "final_y",
        "final_f", "runtime_seconds"
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path

def plot_trajectory(method, function_name, func, path, start, lr, bounds):
    out = ensure_output_dir(method)
    xs = np.linspace(bounds[0][0], bounds[0][1], 250)
    ys = np.linspace(bounds[1][0], bounds[1][1], 250)
    X, Y = np.meshgrid(xs, ys)

    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    finite = np.isfinite(Z)
    if np.any(finite):
        cap = np.percentile(Z[finite], 95)
        Z = np.minimum(Z, cap)

    p = np.asarray(path)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=35)
    plt.plot(p[:, 0], p[:, 1], marker="o", markersize=2, linewidth=1)
    plt.scatter([p[0,0]], [p[0,1]], marker="s", s=70, label="Start")
    plt.scatter([p[-1,0]], [p[-1,1]], marker="*", s=120, label="Final")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{method}: {function_name}\nstart={tuple(start)}, parameter={lr}")
    plt.legend()
    plt.tight_layout()

    name = f"{function_name}_start_{start[0]}_{start[1]}_lr_{lr}.png".replace("-", "neg")
    plt.savefig(out / name, dpi=160)
    plt.close()
