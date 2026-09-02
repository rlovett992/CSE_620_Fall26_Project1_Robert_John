from common import *

def gradient_descent(func, grad, start, learning_rate):
    x = start.astype(float).copy()
    path = [x.copy()]

    start_time = time.perf_counter()
    converged = False

    for iteration in range(1, MAX_ITER + 1):
        new_x = x - learning_rate * grad(x)
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
            for lr in LEARNING_RATES:
                final, path, iterations, converged, runtime = gradient_descent(
                    func, grad, start, lr
                )

                final_value = func(final) if np.all(np.isfinite(final)) else float("nan")

                rows.append({
                    "function": function_name,
                    "start_x": start[0],
                    "start_y": start[1],
                    "learning_rate": lr,
                    "iterations": iterations,
                    "converged": converged,
                    "final_x": final[0],
                    "final_y": final[1],
                    "final_f": final_value,
                    "runtime_seconds": runtime,
                })

                # Keep plots from becoming absurd if the method diverges.
                if np.all(np.isfinite(path[-1])) and np.max(np.abs(path[-1])) < 1e4:
                    plot_trajectory(
                        "gradient_descent", function_name, func, path,
                        start, lr, (xb, yb)
                    )

                print(
                    function_name, tuple(start), lr,
                    "iterations=", iterations,
                    "converged=", converged,
                    "final=", final,
                    "f=", final_value
                )

    result_path = save_results("gradient_descent", rows)
    print(f"\nSaved results to {result_path}")

if __name__ == "__main__":
    main()
