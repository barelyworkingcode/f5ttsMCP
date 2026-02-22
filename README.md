# f5ttsMCP

MCP server for F5-TTS voice synthesis with Apple Silicon MLX acceleration.

## Prerequisites

- macOS with Apple Silicon
- Conda environment `f5tts`:
  ```bash
  conda create -n f5tts python=3.11
  conda activate f5tts
  pip install f5-tts-mlx soundfile mcp
  ```

## Install

```bash
./build.sh
```

Registers with [Relay](https://relay.app):
- **f5tts** -- MCP server (stdio)
- **f5tts-daemon** -- Model server service (autostart, TCP 9998)

## Manual Usage

```bash
# Start the daemon
daemon/daemon_ctl.sh start

# Run MCP server directly
mcp/mcp_wrapper.sh
```

## Project Structure

```
daemon/           Model server (TCP 9998, keeps F5-TTS loaded in memory)
mcp/              MCP server (stdio, connects to daemon)
voices/           Reference audio (.wav) and transcriptions (.txt)
outputs/          Generated audio (runtime, gitignored)
cache/            Processed voice cache (runtime, gitignored)
```

## Voices

You must supply your own voice reference files in the `voices/` folder. Each voice needs a paired `.wav` and `.txt` file with the same name:

```
voices/bob.wav    # Reference audio clip
voices/bob.txt    # Exact transcript of the audio clip
```

**Requirements:**
- **24kHz sample rate** (mandatory -- the model will reject other rates)
- WAV format, mono
- 5-15 seconds of clean speech, minimal background noise
- The `.txt` file must contain the exact words spoken in the `.wav` file

A voice named `primary` is used as the default fallback when no voice is specified.

Also create a `voices/voices.md` file -- the `list_voices` MCP tool returns its contents directly. Use a markdown table with columns for voice name, gender, and notes:

```markdown
| Voice Name | Gender | Notes           |
|------------|--------|-----------------|
| bob        | Male   | Narrator style  |
| alice      | Female | British accent  |
| primary    | Male   | Default voice   |
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `list_voices` | Available voices with metadata |
| `generate_speech` | Text-to-speech with `[VOICE:name]` directives |
| `replay_last_speech` | Replay last generated audio |

## Daemon Options

```
--idle-timeout N    Auto-shutdown after N seconds idle (0 = disabled)
--steps N           Neural ODE steps (default: 8, faster: 4-6)
--method METHOD     Sampling: euler, midpoint, rk4 (default: rk4)
--cfg-strength F    Guidance strength (default: 2.0)
--speed F           Speed factor (default: 1.0)
--quantization N    Model quantization: 4 or 8 bit
```

## Acknowledgments

Built on [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) by Lucas Newman, which provides the MLX-native F5-TTS implementation that makes this project possible. Thank you for the excellent work bringing F5-TTS to Apple Silicon.
