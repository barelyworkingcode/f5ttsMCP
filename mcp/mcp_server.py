#!/usr/bin/env python3
"""
F5-TTS MCP Server
Provides Model Context Protocol integration for F5-TTS voice synthesis system.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from mcp.server import Server
from mcp.types import TextContent, Tool
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("f5-tts-mcp")


class DaemonClient:
    """Client for communicating with the F5-TTS daemon."""

    def __init__(self, host: str = 'localhost', port: int = 9998, timeout: float = 300):
        self.host = host
        self.port = port
        self.timeout = timeout

    def is_daemon_running(self) -> bool:
        """Check if the daemon is running."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((self.host, self.port))
            sock.close()
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False

    def send_streaming_batch(self, batch_items: List[Dict]) -> Generator[Dict, None, None]:
        """Send a streaming batch request and yield results as they arrive."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect((self.host, self.port))

            # Send batch request with stream flag
            request = {
                'batch': batch_items,
                'stream': True
            }
            sock.send(json.dumps(request).encode('utf-8'))

            # Read newline-delimited JSON responses
            buffer = ""
            while True:
                try:
                    data = sock.recv(4096).decode('utf-8')
                    if not data:
                        break

                    buffer += data

                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            try:
                                result = json.loads(line)
                                yield result

                                # Stop if we got the completion message
                                if result.get('type') == 'complete':
                                    return
                                if result.get('type') == 'error':
                                    return
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON in stream: {e}")

                except socket.timeout:
                    logger.error("Timeout waiting for daemon response")
                    break

        except ConnectionRefusedError:
            logger.error("Daemon not running")
            yield {'type': 'error', 'success': False, 'error': 'Daemon not running'}
        except Exception as e:
            logger.error(f"Daemon communication error: {e}")
            yield {'type': 'error', 'success': False, 'error': str(e)}
        finally:
            sock.close()

class F5TTSServer:
    def __init__(self):
        self.server = Server("f5-tts")
        self.base_path = Path(__file__).parent.parent
        self.voices_path = self.base_path / "voices"
        self.outputs_path = self.base_path / "outputs"
        self.daemon_client = DaemonClient()

        # Ensure outputs directory exists
        self.outputs_path.mkdir(exist_ok=True)

        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP request handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available F5-TTS tools."""
            return [
                Tool(
                    name="list_voices",
                    description="Get list of available F5-TTS voices with gender information. Before calling this tool, acknowledge that you are retrieving the voice list.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    name="generate_speech",
                    description="Generate speech from text using F5-TTS with voice assignments. Before calling this tool, confirm what text will be converted to speech and which voices will be used. Text should include [VOICE:voice_name] directives for multi-voice content.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to convert to speech. Use [VOICE:voice_name] directives to assign voices to different chunks."
                            },
                            "play_immediately": {
                                "type": "boolean",
                                "description": "Whether to play the generated audio immediately",
                                "default": True
                            }
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    name="replay_last_speech",
                    description="Replay the last generated speech audio chunks in sequence. Before calling this tool, confirm that you are replaying the most recent speech generation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "delay_seconds": {
                                "type": "number",
                                "description": "Delay between audio chunks in seconds",
                                "default": 0.25
                            }
                        },
                        "additionalProperties": False,
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "list_voices":
                    return await self._list_voices()
                elif name == "generate_speech":
                    return await self._generate_speech(
                        text=arguments["text"],
                        play_immediately=arguments.get("play_immediately", True)
                    )
                elif name == "replay_last_speech":
                    return await self._replay_last_speech(
                        delay_seconds=arguments.get("delay_seconds", 0.25)
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _parse_voice_chunks(self, text: str) -> List[Dict[str, str]]:
        """Parse text with [VOICE:name] directives into chunks for batch processing."""
        # Pattern to match [VOICE:name] followed by text
        pattern = r'\[VOICE:([^\]]+)\]\s*([^[]*?)(?=\[VOICE:|$)'
        matches = re.findall(pattern, text)

        if not matches:
            # No voice directives found, use default voice
            return [{'voice': 'primary', 'text': text.strip()}]

        chunks = []
        for voice_name, voice_text in matches:
            voice_text = voice_text.strip()
            if voice_text:
                chunks.append({'voice': voice_name, 'text': voice_text})

        return chunks

    def _get_voice_file(self, voice_name: str) -> tuple[str, Optional[str]]:
        """Get voice file path and ref_text for a voice name."""
        voice_file = self.voices_path / f"{voice_name}.wav"
        text_file = self.voices_path / f"{voice_name}.txt"

        # Fall back to primary if voice doesn't exist
        if not voice_file.exists():
            voice_file = self.voices_path / "primary.wav"
            text_file = self.voices_path / "primary.txt"

        ref_text = None
        if text_file.exists():
            ref_text = text_file.read_text().strip()

        return str(voice_file), ref_text

    def _play_audio_file(self, audio_file: str) -> None:
        """Play an audio file using system audio player."""
        if not os.path.exists(audio_file):
            logger.warning(f"Audio file not found: {audio_file}")
            return

        try:
            # Use afplay on macOS
            subprocess.run(
                ["afplay", audio_file],
                check=True,
                capture_output=True
            )
        except FileNotFoundError:
            # Fall back to sox/play
            try:
                subprocess.run(
                    ["play", audio_file],
                    check=True,
                    capture_output=True,
                    stderr=subprocess.DEVNULL
                )
            except FileNotFoundError:
                logger.warning("No audio player found (afplay or play)")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Audio playback failed: {e}")

    async def _list_voices(self) -> List[TextContent]:
        """Read and return the list of available voices."""
        try:
            voices_md_path = self.voices_path / "voices.md"
            if not voices_md_path.exists():
                return [TextContent(type="text", text="Error: voices.md not found")]

            with open(voices_md_path, 'r') as f:
                voices_content = f.read()

            return [TextContent(type="text", text=voices_content)]
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return [TextContent(type="text", text=f"Error reading voices: {str(e)}")]

    async def _generate_speech(self, text: str, play_immediately: bool = True) -> List[TextContent]:
        """Generate speech from text using direct daemon communication with streaming."""
        start_time = time.time()

        # Check if daemon is running
        if not self.daemon_client.is_daemon_running():
            return [TextContent(
                type="text",
                text="Error: F5-TTS daemon is not running. Start it with: daemon/daemon_ctl.sh start"
            )]

        # Parse text into voice chunks
        chunks = self._parse_voice_chunks(text)
        logger.info(f"Parsed {len(chunks)} voice chunks")

        # Clean up old chunk directories
        for old_dir in self.outputs_path.glob("chunks_*"):
            if old_dir.is_dir():
                shutil.rmtree(old_dir, ignore_errors=True)

        # Create output directory for this batch
        chunk_dir = self.outputs_path / "chunks_mcp"
        chunk_dir.mkdir(exist_ok=True)

        # Build batch request
        batch_items = []
        for i, chunk in enumerate(chunks):
            voice_file, ref_text = self._get_voice_file(chunk['voice'])
            output_file = str(chunk_dir / f"audio_{i+1:03d}.wav")

            item = {
                'text': chunk['text'],
                'voice_file': voice_file,
                'output_file': output_file,
            }
            if ref_text:
                item['ref_text'] = ref_text
            else:
                # Request transcription if no ref_text exists
                text_file = self.voices_path / f"{chunk['voice']}.txt"
                item['save_ref_text'] = str(text_file)

            batch_items.append(item)

        # Send streaming batch request and process results
        successful = 0
        total = len(batch_items)
        first_audio_time = None
        total_audio_duration = 0.0

        logger.info(f"Sending streaming batch of {total} items to daemon")

        for result in self.daemon_client.send_streaming_batch(batch_items):
            result_type = result.get('type')

            if result_type == 'error':
                error_msg = result.get('error', 'Unknown daemon error')
                logger.error(f"Daemon error: {error_msg}")
                return [TextContent(type="text", text=f"Error: {error_msg}")]

            elif result_type == 'chunk':
                index = result.get('index', 0)
                success = result.get('success', False)
                output_file = result.get('output_file', '')
                timing = result.get('timing', {})

                if success:
                    successful += 1
                    if timing:
                        total_audio_duration += timing.get('audio_duration', 0)

                    # Record time to first audio
                    if first_audio_time is None:
                        first_audio_time = time.time() - start_time
                        logger.info(f"Time to first audio: {first_audio_time:.2f}s")

                    # Play audio immediately if requested
                    if play_immediately and output_file:
                        logger.info(f"Playing chunk {index + 1}/{total}")
                        self._play_audio_file(output_file)
                else:
                    logger.warning(f"Chunk {index + 1} failed: {result.get('result', 'Unknown error')}")

            elif result_type == 'complete':
                logger.info(f"Batch complete: {result.get('successful_items')}/{result.get('total_items')}")

        total_time = time.time() - start_time

        # Build response summary
        response_parts = ["Speech generation completed!"]
        response_parts.append(f"Chunks: {successful}/{total} successful")
        if first_audio_time is not None:
            response_parts.append(f"Time to first audio: {first_audio_time:.1f}s")
        response_parts.append(f"Total time: {total_time:.1f}s")
        if total_audio_duration > 0:
            response_parts.append(f"Audio duration: {total_audio_duration:.1f}s")
        response_parts.append(f"Output: {chunk_dir}")

        return [TextContent(type="text", text="\n".join(response_parts))]

    async def _replay_last_speech(self, delay_seconds: float = 0.25) -> List[TextContent]:
        """Replay the last generated speech chunks."""
        try:
            # Find the most recent chunk directory
            chunk_dirs = [d for d in self.outputs_path.iterdir()
                         if d.is_dir() and d.name.startswith('chunks_')]

            if not chunk_dirs:
                return [TextContent(type="text", text="No previous speech generation found to replay")]

            # Get the most recent directory (by modification time)
            most_recent = max(chunk_dirs, key=lambda d: d.stat().st_mtime)
            basename = most_recent.name.replace('chunks_', '')

            # Run replay_chunks.sh (colocated in mcp/)
            replay_script = Path(__file__).parent / "replay_chunks.sh"
            cmd = [str(replay_script), basename, str(delay_seconds)]

            logger.info(f"Running replay command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout for replay
            )

            if result.returncode != 0:
                error_msg = f"replay_chunks.sh failed with return code {result.returncode}\nStderr: {result.stderr}"
                logger.error(error_msg)
                return [TextContent(type="text", text=error_msg)]

            # Extract summary from output
            output_lines = result.stdout.strip().split('\n')
            summary_lines = []
            for line in reversed(output_lines):
                if 'complete' in line.lower() or 'chunks' in line.lower() or 'time' in line.lower():
                    summary_lines.insert(0, line)
                if 'complete' in line.lower():
                    break

            response = f"Replay completed for: {basename}\n\n"
            if summary_lines:
                response += "\n".join(summary_lines)
            else:
                response += result.stdout

            return [TextContent(type="text", text=response)]

        except subprocess.TimeoutExpired:
            return [TextContent(type="text", text="Error: Replay timed out after 2 minutes")]
        except Exception as e:
            logger.error(f"Error replaying speech: {e}")
            return [TextContent(type="text", text=f"Error replaying speech: {str(e)}")]

    async def run(self):
        """Run the MCP server."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

async def main():
    """Main entry point."""
    server = F5TTSServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
