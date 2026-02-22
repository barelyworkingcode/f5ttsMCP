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
