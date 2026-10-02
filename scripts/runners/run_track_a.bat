@echo off
cd /d "C:\Projects\FYP Project"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
pip install "accelerate>=1.1.0" -q
python scripts/train_arabert_regression.py > output\track_a_train_stdout.log 2> output\track_a_train_stderr.log
exit /b %ERRORLEVEL%
