#!/bin/bash
# F5-TTS Daemon Control Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DAEMON_PID_FILE="/tmp/f5tts_daemon.pid"
DAEMON_LOG_FILE="/tmp/f5tts_daemon.log"
DAEMON_SCRIPT="$SCRIPT_DIR/f5_daemon.py"

start_daemon() {
    if [ -f "$DAEMON_PID_FILE" ]; then
        PID=$(cat "$DAEMON_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Daemon already running with PID $PID"
            return 1
        else
            echo "Removing stale PID file"
            rm -f "$DAEMON_PID_FILE"
        fi
    fi

    echo "Starting F5-TTS daemon..."

    # Activate conda environment
    source ~/miniconda3/bin/activate f5tts

    # Set environment variables
    export PYTHONHASHSEED=random

    # Start daemon in background (CWD = project root for relative output paths)
    cd "$PROJECT_DIR"
    nohup python "$DAEMON_SCRIPT" > "$DAEMON_LOG_FILE" 2>&1 &
    DAEMON_PID=$!

    # Save PID
    echo "$DAEMON_PID" > "$DAEMON_PID_FILE"

    # Wait a moment and check if it's still running
    sleep 2
    if ps -p "$DAEMON_PID" > /dev/null 2>&1; then
        echo "Daemon started with PID $DAEMON_PID"
        echo "Log file: $DAEMON_LOG_FILE"
        return 0
    else
        echo "Daemon failed to start. Check log: $DAEMON_LOG_FILE"
        rm -f "$DAEMON_PID_FILE"
        return 1
    fi
}

stop_daemon() {
    if [ ! -f "$DAEMON_PID_FILE" ]; then
        echo "No daemon PID file found"
        return 1
    fi

    PID=$(cat "$DAEMON_PID_FILE")

    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping daemon (PID $PID)..."
        kill "$PID"

        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! ps -p "$PID" > /dev/null 2>&1; then
                echo "Daemon stopped"
                rm -f "$DAEMON_PID_FILE"
                return 0
            fi
            sleep 1
        done

        # Force kill if still running
        echo "Force killing daemon..."
        kill -9 "$PID" 2>/dev/null
        rm -f "$DAEMON_PID_FILE"
        echo "Daemon force stopped"
    else
        echo "Daemon not running"
        rm -f "$DAEMON_PID_FILE"
    fi
}

status_daemon() {
    if [ -f "$DAEMON_PID_FILE" ]; then
        PID=$(cat "$DAEMON_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Daemon running with PID $PID"
            echo "Log file: $DAEMON_LOG_FILE"

            # Test connection
            if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost', 9998)); s.close()" 2>/dev/null; then
                echo "Daemon responding on port 9998"
            else
                echo "Daemon not responding on port 9998"
            fi
            return 0
        else
            echo "Daemon not running (stale PID file)"
            rm -f "$DAEMON_PID_FILE"
            return 1
        fi
    else
        echo "Daemon not running"
        return 1
    fi
}

show_log() {
    if [ -f "$DAEMON_LOG_FILE" ]; then
        echo "Last 20 lines of daemon log:"
        echo "================================"
        tail -20 "$DAEMON_LOG_FILE"
    else
        echo "No log file found"
    fi
}

case "$1" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        stop_daemon
        sleep 1
        start_daemon
        ;;
    status)
        status_daemon
        ;;
    log)
        show_log
        ;;
    *)
        echo "F5-TTS Daemon Control"
        echo "Usage: $0 {start|stop|restart|status|log}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the daemon"
        echo "  stop    - Stop the daemon"
        echo "  restart - Restart the daemon"
        echo "  status  - Show daemon status"
        echo "  log     - Show recent log entries"
        exit 1
        ;;
esac
