"""Self-contained: install accelerate, then train AraBERT regression."""
import subprocess, sys, os

os.chdir(r"C:\Projects\FYP Project")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Step 1: Install accelerate
print("Installing accelerate...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "accelerate>=1.1.0", "-q"],
    capture_output=True,
)

# Step 2: Verify
try:
    import accelerate
    print(f"accelerate {accelerate.__version__} ready")
except ImportError:
    print("FAILED to install accelerate")
    sys.exit(1)

# Step 3: Run training
print("Starting Track A training...")
sys.stdout.flush()
sys.stderr.flush()
result = subprocess.run(
    [sys.executable, "scripts/train_arabert_regression.py"],
    capture_output=False,
)
sys.exit(result.returncode)
