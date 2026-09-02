import subprocess
import sys

scripts = [
    "gradient_descent.py",
    "newton.py",
    "adagrad.py",
    "adam.py",
]

for script in scripts:
    print("\n" + "=" * 70)
    print(f"RUNNING {script}")
    print("=" * 70)
    subprocess.run([sys.executable, script], check=True)

print("\nAll simulations completed.")
