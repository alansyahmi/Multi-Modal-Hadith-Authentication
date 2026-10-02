# Launch Track A AraBERT training
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = 1
$env:PYTHONIOENCODING = "utf-8"

Write-Output "Installing accelerate..."
pip install "accelerate>=1.1.0" -q

Write-Output "Starting training..."
python scripts/train_arabert_regression.py
