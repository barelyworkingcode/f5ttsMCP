#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Register with Relay (best-effort)
RELAY="/Applications/Relay.app/Contents/MacOS/relay"
if [ -x "$RELAY" ]; then
    "$RELAY" mcp register --name f5tts --command "$SCRIPT_DIR/mcp/mcp_wrapper.sh"
    "$RELAY" service register --name f5tts-daemon --command "$SCRIPT_DIR/daemon/daemon_wrapper.sh" --autostart
    echo "Registered f5tts MCP and f5tts-daemon service with Relay"
else
    echo "Relay not found at $RELAY, skipping registration"
fi
