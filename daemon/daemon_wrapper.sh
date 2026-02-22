#!/bin/bash
# Wrapper script to activate conda environment before running F5-TTS daemon

# Activate conda environment
source ~/miniconda3/bin/activate f5tts

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONHASHSEED=random

# Run the daemon with the conda environment's Python
exec python "$SCRIPT_DIR/f5_daemon.py" --idle-timeout 0 "$@"
