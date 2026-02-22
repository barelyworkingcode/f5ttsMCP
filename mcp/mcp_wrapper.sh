#!/bin/bash
# MCP wrapper script to activate conda environment before running F5-TTS MCP server

# Activate conda environment
source ~/miniconda3/bin/activate f5tts

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the MCP server with the conda environment's Python
exec python "$SCRIPT_DIR/mcp_server.py" "$@"
