from common import *

# These values act as damped Newton step sizes.
DAMPING_VALUES = [0.25, 0.5, 1.0]

def newtons_method(func, grad, hess, start, damping):
    x = start.astype(float).copy()
    path = [x.copy()]

    start_time = time.perf_counter()
    converged = False

    for iteration in range(1, MAX_ITER + 1):
        g = grad(x)
        H = hess(x)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Pseudoinverse allows the experiment to continue if Hessian is singular.
            step = np.linalg.pinv(H) @ g

        new_x = x - damping * step
        path.append(new_x.copy())

        if not np.all(np.isfinite(new_x)):
            x = new_x
            break

        if np.linalg.norm(new_x - x) < TOL:
            x = new_x
            converged = True
            break

        x = new_x

    runtime = time.perf_counter() - start_time
    return x, path, iteration, converged, runtime

def main():
    rows = []

    for function_name, (func, grad, hess, xb, yb) in FUNCTIONS.items():
        for start in INITIAL_POINTS:
            for damping in DAMPING_VALUES:
                final, path, iterations, converged, runtime = newtons_method(
                    func, grad, hess, start, damping
                )

                final_value = func(final) if np.all(np.isfinite(final)) else float("nan")

                rows.append({
                    "function": function_name,
                    "start_x": start[0],
                    "start_y": start[1],
                    "learning_rate": damping,
                    "iterations": iterations,
                    "converged": converged,
                    "final_x": final[0],
                    "final_y": final[1],
                    "final_f": final_value,
                    "runtime_seconds": runtime,
                })

                if np.all(np.isfinite(path[-1])) and np.max(np.abs(path[-1])) < 1e4:
                    plot_trajectory(
                        "newton", function_name, func, path,
                        start, damping, (xb, yb)
                    )

                print(
                    function_name, tuple(start), "damping=", damping,
                    "iterations=", iterations,
                    "converged=", converged,
                    "final=", final,
                    "f=", final_value
                )

    result_path = save_results("newton", rows)
    print(f"\nSaved results to {result_path}")

if __name__ == "__main__":
    main()
