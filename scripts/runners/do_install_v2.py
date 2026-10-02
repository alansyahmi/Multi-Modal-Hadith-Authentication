"""Install accelerate using pip's internal API (no subprocess)."""
import pip._internal.cli.main as pip_main
import sys

sys.argv = ["pip", "install", "accelerate>=1.1.0"]
try:
    pip_main.main()
    print("Installation complete")
except SystemExit:
    pass

# Verify
try:
    import accelerate
    print(f"accelerate {accelerate.__version__} installed successfully")
except ImportError:
    print("accelerate not found after install")
    sys.exit(1)
