#!/bin/bash
# F5-TTS Chunk Replay Script - replays previously generated audio chunks
# Usage: ./replay_chunks.sh basename [delay_seconds]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ $# -eq 0 ]; then
    echo "F5-TTS Chunk Replay"
    echo "Usage: $0 basename [delay_seconds]"
    echo ""
    echo "Examples:"
    echo "  $0 document_screenplay"
    echo "  $0 document_screenplay 0.5"
    echo ""
    echo "Default delay between chunks: 0.25 seconds"
    echo ""
    echo "Available chunk sets:"
    ls -d "$PROJECT_DIR"/outputs/chunks_* 2>/dev/null | sed "s|$PROJECT_DIR/outputs/chunks_||" | sed 's/^/  /'
    exit 1
fi

BASENAME="$1"
DELAY="${2:-0.25}"
CHUNK_DIR="$PROJECT_DIR/outputs/chunks_${BASENAME}"

# Check if chunk directory exists
if [ ! -d "$CHUNK_DIR" ]; then
    echo "Chunk directory not found: $CHUNK_DIR"
    echo ""
    echo "Available chunk sets:"
    ls -d "$PROJECT_DIR"/outputs/chunks_* 2>/dev/null | sed "s|$PROJECT_DIR/outputs/chunks_||" | sed 's/^/  /'
    exit 1
fi

# Get list of audio files in order
AUDIO_FILES=$(ls "$CHUNK_DIR"/audio_*.wav 2>/dev/null | sort)
CHUNK_COUNT=$(echo "$AUDIO_FILES" | wc -l)

if [ $CHUNK_COUNT -eq 0 ]; then
    echo "No audio chunks found in: $CHUNK_DIR"
    exit 1
fi

echo "Replaying $CHUNK_COUNT audio chunks from: $CHUNK_DIR"
echo "Delay between chunks: ${DELAY}s"
echo ""

START_TIME=$(date +%s.%N)

# Function to play audio file
play_audio() {
    local audio_file="$1"
    local chunk_num=$(basename "$audio_file" .wav | sed 's/audio_0*//')

    echo "Playing chunk $chunk_num: $(basename "$audio_file")"

    if command -v afplay &> /dev/null; then
        afplay "$audio_file"
    elif command -v sox &> /dev/null && command -v play &> /dev/null; then
        play "$audio_file" 2>/dev/null
    else
        echo "No audio player found (afplay or sox/play)"
        return 1
    fi
}

# Play all chunks with delays
chunk_num=1
while IFS= read -r audio_file; do
    if [ -n "$audio_file" ]; then
        play_audio "$audio_file"

        # Add delay between chunks (except after the last one)
        if [ $chunk_num -lt $CHUNK_COUNT ]; then
            sleep "$DELAY"
        fi

        chunk_num=$((chunk_num + 1))
    fi
done <<< "$AUDIO_FILES"

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc -l)

echo ""
echo "Replay complete!"
echo "Total replay time: ${TOTAL_TIME%.*}s for $CHUNK_COUNT chunks"
echo "Chunks remain saved in: $CHUNK_DIR"
