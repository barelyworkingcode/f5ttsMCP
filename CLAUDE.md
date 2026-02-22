# f5ttsMCP

F5-TTS voice synthesis via MCP. Two-process architecture: a daemon (MLX model server on TCP 9998) and an MCP server that talks to it.

## Architecture

- `daemon/` -- F5-TTS MLX model server. Keeps model loaded in memory, accepts TCP requests on port 9998. Managed by Relay as a service.
- `mcp/` -- MCP server (stdio). Parses voice directives, sends batch requests to daemon, plays audio. Managed by Relay as an MCP.
- `voices/` -- Voice reference audio (.wav) and transcription (.txt) files.
- `outputs/` -- Generated audio chunks. Gitignored, created at runtime.
- `cache/voice_audio/` -- Processed voice audio cache. Gitignored, created at runtime.

## Setup

Requires conda env `f5tts` with `f5-tts-mlx`, `soundfile`, `mcp` packages.

```bash
./build.sh  # Registers MCP + daemon service with Relay
```

Manual start (without Relay):
```bash
daemon/daemon_ctl.sh start   # Start daemon
python mcp/mcp_server.py     # Run MCP server
```

## MCP Tools

- `list_voices` -- Returns available voices from voices/voices.md
- `generate_speech` -- Text-to-speech with `[VOICE:name]` directives for multi-voice
- `replay_last_speech` -- Replay last generated audio chunks

## Key paths

All paths resolve from `__file__`, not CWD. The daemon cache dir, MCP base path, and shell scripts all use script-relative resolution.

## Daemon flags

`--idle-timeout 0` disables auto-shutdown (default in daemon_wrapper.sh for Relay). `--steps`, `--method`, `--cfg-strength`, `--speed` tune generation performance.
