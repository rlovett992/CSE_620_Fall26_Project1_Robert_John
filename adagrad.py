from common import *

def adagrad(func, grad, start, learning_rate, epsilon=1e-8):
    x = start.astype(float).copy()
    accumulated_sq_grad = np.zeros_like(x)
    path = [x.copy()]

    start_time = time.perf_counter()
    converged = False

    for iteration in range(1, MAX_ITER + 1):
        g = grad(x)
        accumulated_sq_grad += g * g
        adjusted_step = learning_rate * g / (np.sqrt(accumulated_sq_grad) + epsilon)
        new_x = x - adjusted_step
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
                final, path, iterations, converged, runtime = adagrad(
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

                plot_trajectory(
                    "adagrad", function_name, func, path,
                    start, lr, (xb, yb)
                )

                print(
                    function_name, tuple(start), lr,
                    "iterations=", iterations,
                    "converged=", converged,
                    "final=", final,
                    "f=", final_value
                )

    result_path = save_results("adagrad", rows)
    print(f"\nSaved results to {result_path}")

if __name__ == "__main__":
    main()
