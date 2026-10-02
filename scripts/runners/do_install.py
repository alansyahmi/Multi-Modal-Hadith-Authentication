"""Install accelerate package."""
import subprocess, sys
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "accelerate>=1.1.0"],
    capture_output=True, text=True,
)
print(r.stdout)
print(r.stderr)
print(f"Return code: {r.returncode}")
