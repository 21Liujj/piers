@echo off
REM piers-core CPU setup (Windows/WSL). No models, no CUDA required.
python -m pip install -r requirements-core.txt
python -m pip install -e .
piers_ljj doctor
piers_ljj smoke
piers_ljj validate-data
python -m pytest tests/piers -q
