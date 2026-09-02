Gradient Descent Variants - Project Scripts

Files
-----
common.py              Objective functions, gradients, Hessians, plotting helpers
gradient_descent.py    Fixed-learning-rate Gradient Descent
newton.py              Newton's Method with damping
adagrad.py             AdaGrad
adam.py                Adam
run_all.py             Runs all four simulations

Install dependencies
--------------------
python -m pip install -r requirements.txt

Run one method
--------------
python gradient_descent.py
python newton.py
python adagrad.py
python adam.py

Run everything
--------------
python run_all.py

Output
------
Each optimizer creates:
output/<optimizer>/results.csv
output/<optimizer>/*.png

The CSV records the function, initial point, parameter value, number of
iterations, convergence status, final point, final function value, and runtime.

The scripts use:
- initial points: (-2, 2), (0.5, -1.5), (3, 3)
- parameter values: 0.001, 0.01, 0.1 for GD/AdaGrad/Adam
- damping values: 0.25, 0.5, 1.0 for Newton
- convergence tolerance: ||x_(k+1) - x_k|| < 1e-6
- maximum iterations: 2000
