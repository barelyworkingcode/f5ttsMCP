#!/usr/bin/env python3
"""
F5-TTS Daemon Server
Keeps F5-TTS model loaded in memory with Apple Silicon MLX acceleration
"""
import os
import json
import socket
import struct
import subprocess
import threading
import warnings
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import mlx.core as mx

from f5_tts_mlx.generate import generate as f5_generate, SAMPLE_RATE
from f5_tts_mlx.utils import convert_char_to_pinyin
from f5_tts_mlx.cfm import F5TTS

# Suppress warnings
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed")
warnings.filterwarnings("ignore", message="resource_tracker: process died unexpectedly")

class F5TTSDaemon:
    def __init__(self, host='localhost', port=9998, idle_timeout=900, quantization_bits=None,
                 steps=8, method="rk4", cfg_strength=2.0, speed=1.0):
        self.host = host
        self.port = port
        self.model = None
        self.running = False
        self.sock = None
        self.idle_timeout = idle_timeout
        self.last_activity = None  # Will be set after daemon is ready
        self.activity_lock = threading.Lock()
        self.quantization_bits = quantization_bits  # 4, 8, or None for full precision

        # Generation parameters for performance tuning
        self.steps = steps  # Neural ODE steps (default: 8, faster: 4-6)
        self.method = method  # Sampling method (rk4, euler, midpoint)
        self.cfg_strength = cfg_strength  # Classifier-free guidance strength
        self.speed = speed  # Speed factor for duration estimation

        # Voice audio processing cache (stores batch-ready expanded tensors)
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_dir = Path(__file__).parent.parent / "cache" / "voice_audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Pinyin conversion cache for ref_text (keyed by raw ref_text string)
        self.pinyin_cache = {}
        self.pinyin_cache_lock = threading.Lock()

        # Thread pool for async file I/O
        self.io_executor = ThreadPoolExecutor(max_workers=2)

    def get_voice_cache_key(self, voice_file):
        """Generate cache key for voice file based on file path and modification time"""
        try:
            stat = os.stat(voice_file)
            file_hash = f"{voice_file}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(file_hash.encode()).hexdigest()
        except OSError:
            return None

    def get_cached_audio(self, voice_file):
        """Get cached processed audio for voice file if available.

        Returns an already-expanded tensor (batch dim included) ready for f5tts.sample().
        """
        cache_key = self.get_voice_cache_key(voice_file)
        if not cache_key:
            return None

        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.audio_cache:
                return self.audio_cache[cache_key]

            # Check disk cache (.npy format)
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                try:
                    audio_array = np.load(cache_file)
                    mx_audio = mx.array(audio_array)
                    self.audio_cache[cache_key] = mx_audio
                    return mx_audio
                except Exception:
                    pass

            # Legacy .npz fallback -- migrate on read
            legacy_file = self.cache_dir / f"{cache_key}.npz"
            if legacy_file.exists():
                try:
                    data = np.load(legacy_file)
                    audio_array = data['audio']
                    # Legacy cache stored raw audio; expand to batch dim
                    if audio_array.ndim == 1:
                        audio_array = np.expand_dims(audio_array, axis=0)
                    mx_audio = mx.array(audio_array)
                    # Re-save as .npy with expanded dims and remove legacy
                    np.save(cache_file, audio_array)
                    legacy_file.unlink(missing_ok=True)
                    self.audio_cache[cache_key] = mx_audio
                    return mx_audio
                except Exception:
                    pass

        return None

    def cache_audio(self, voice_file, processed_audio):
        """Cache processed audio for voice file (stores expanded tensor)"""
        cache_key = self.get_voice_cache_key(voice_file)
        if not cache_key or processed_audio is None:
            return

        with self.cache_lock:
            # Store in memory cache
            self.audio_cache[cache_key] = processed_audio

            # Store in disk cache (uncompressed for faster writes)
            cache_file = self.cache_dir / f"{cache_key}.npy"
            try:
                audio_np = np.array(processed_audio)
                np.save(cache_file, audio_np)
            except Exception as e:
                print(f"Failed to cache processed audio: {e}")

    def get_ref_text_pinyin(self, ref_text):
        """Get pinyin-converted ref_text, using cache to avoid recomputation."""
        with self.pinyin_cache_lock:
            if ref_text in self.pinyin_cache:
                return self.pinyin_cache[ref_text]

        # Compute outside the lock (pure function, safe for concurrent calls)
        pinyin = convert_char_to_pinyin([ref_text])[0]

        with self.pinyin_cache_lock:
            self.pinyin_cache[ref_text] = pinyin
        return pinyin

    def update_activity(self):
        """Update last activity timestamp"""
        with self.activity_lock:
            self.last_activity = time.time()

    def check_idle_timeout(self):
        """Check if daemon should terminate due to inactivity"""
        if self.idle_timeout <= 0:
            return False
        with self.activity_lock:
            if self.last_activity is None:
                return False  # Don't timeout during startup
            idle_time = time.time() - self.last_activity
            return idle_time > self.idle_timeout

    def idle_monitor(self):
        """Monitor thread to check for idle timeout"""
        if self.idle_timeout <= 0:
            return
        while self.running:
            if self.check_idle_timeout():
                print(f"Daemon idle for {self.idle_timeout//60} minutes, shutting down...")
                self.running = False
                break
            time.sleep(30)  # Check every 30 seconds

    def load_model(self):
        """Load F5-TTS model once at startup"""
        try:
            if self.quantization_bits:
                print(f"Loading F5-TTS model with {self.quantization_bits}-bit quantization...")
                print(f"Expected memory savings: ~{40 if self.quantization_bits == 8 else 50}%")
            else:
                print("Loading F5-TTS model (full precision)...")

            # Store both the generate function and the model instance with quantization
            self.model = f5_generate
            self.f5tts_instance = F5TTS.from_pretrained(
                "lucasnewman/f5-tts-mlx",
                quantization_bits=self.quantization_bits
            )

            if self.quantization_bits:
                print(f"F5-TTS model loaded with {self.quantization_bits}-bit quantization!")
                print(f"Expected performance improvement: faster inference, lower memory usage")
            else:
                print("F5-TTS model loaded (full precision)!")

            # Show generation parameters
            print(f"Generation settings: {self.steps} steps, {self.method} method, CFG {self.cfg_strength}, speed {self.speed}x")
            if self.steps < 8:
                print(f"Performance mode: {self.steps} steps (vs default 8) for faster generation")
            if self.method != "rk4":
                print(f"Fast sampling: {self.method} method (vs default rk4) for speed optimization")

            # Warmup: run a dummy generation to force MLX compilation
            self._warmup_model()

            return True
        except Exception as e:
            print(f"Error loading F5-TTS: {e}")
            return False

    def _warmup_model(self):
        """Run a short dummy generation to trigger MLX lazy compilation."""
        try:
            print("Warming up model (first-run compilation)...")
            warmup_start = time.time()

            # Create a tiny synthetic reference audio (0.5s of silence at 24kHz)
            dummy_audio = mx.zeros((1, SAMPLE_RATE // 2))
            dummy_text = convert_char_to_pinyin(["warmup. hello"])

            self.f5tts_instance.sample(
                dummy_audio,
                text=dummy_text,
                duration=SAMPLE_RATE,  # Very short duration
                steps=2,
                method="euler",
                speed=1.0,
                cfg_strength=1.0,
                sway_sampling_coef=-1.0,
                seed=42,
            )

            warmup_time = time.time() - warmup_start
            print(f"Model warmup complete ({warmup_time:.1f}s)")
        except Exception as e:
            print(f"Model warmup failed (non-fatal): {e}")

    def generate_speech(self, text, voice_file, output_file="outputs/daemon_output.wav", ref_text=None, save_ref_text=None):
        """Generate speech using the loaded model"""
        try:
            if not self.model:
                return False, "Model not loaded", None

            if not os.path.exists(voice_file):
                return False, f"Voice file not found: {voice_file}", None

            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir:  # Only create directory if there is one
                os.makedirs(output_dir, exist_ok=True)

            # Handle reference text
            if not ref_text:
                if save_ref_text:
                    # Need to transcribe
                    try:
                        print("Auto-transcribing reference audio...")
                        from f5_tts.infer.utils_infer import transcribe
                        transcribed_text = transcribe(voice_file)
                        ref_text = transcribed_text
                        print(f"Transcribed: '{ref_text}'")

                        # Save transcribed text to file if requested
                        with open(save_ref_text, 'w', encoding='utf-8') as f:
                            f.write(transcribed_text)
                        print(f"Saved transcription to: {save_ref_text}")

                    except Exception as e:
                        print(f"Transcription failed, using default: {e}")
                        ref_text = "Hello, this is my voice speaking clearly and naturally. I am recording this sample for voice cloning purposes."
                else:
                    # Use default
                    ref_text = "Hello, this is my voice speaking clearly and naturally. I am recording this sample for voice cloning purposes."

            # Try to get cached processed audio first (already expanded with batch dim)
            cache_start = time.time()
            audio_expanded = self.get_cached_audio(voice_file)
            cache_time = time.time() - cache_start

            if audio_expanded is not None:
                print(f"CACHE HIT: Loaded processed audio in {cache_time*1000:.2f}ms for: {os.path.basename(voice_file)}")
                print(f"Audio shape: {audio_expanded.shape}, skipping file I/O and processing")
            else:
                print(f"CACHE MISS: No cached audio found for: {os.path.basename(voice_file)}")
                # Load and process audio file
                load_start = time.time()
                print(f"Loading and processing audio file...")

                # Load reference audio
                audio, sr = sf.read(voice_file)
                if sr != 24000:  # F5-TTS requires 24kHz
                    raise ValueError(f"Reference audio must have a sample rate of 24kHz, got {sr}")
                load_time = time.time() - load_start

                # Convert to MLX array, normalize, and expand for batch dim
                process_start = time.time()
                audio = mx.array(audio)
                TARGET_RMS = 0.1
                rms = mx.sqrt(mx.mean(mx.square(audio)))
                if rms < TARGET_RMS:
                    audio = audio * TARGET_RMS / rms
                audio_expanded = mx.expand_dims(audio, axis=0)
                process_time = time.time() - process_start

                # Cache the expanded audio tensor
                cache_save_start = time.time()
                self.cache_audio(voice_file, audio_expanded)
                cache_save_time = time.time() - cache_save_start

                total_processing = load_time + process_time + cache_save_time
                print(f"CACHED: Load {load_time*1000:.1f}ms + Process {process_time*1000:.1f}ms + Save {cache_save_time*1000:.1f}ms = {total_processing*1000:.1f}ms total")

            # Length of raw audio (without batch dim) for trimming generated output
            ref_audio_len = audio_expanded.shape[1]

            transcribed_text = None

            print(f"Generating: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            start_time = time.time()

            # Generate speech using MLX with cached processed audio
            print("Using optimized generation with cached audio")
            try:
                f5tts = self.f5tts_instance

                # Use cached pinyin for ref_text, only convert generation text fresh
                ref_pinyin = self.get_ref_text_pinyin(ref_text)
                gen_pinyin = convert_char_to_pinyin([text])[0]
                generation_text_processed = [ref_pinyin + " " + gen_pinyin]

                # Generate using cached expanded audio tensor
                wave, _ = f5tts.sample(
                    audio_expanded,
                    text=generation_text_processed,
                    duration=None,  # Let it estimate
                    steps=self.steps,
                    method=self.method,
                    speed=self.speed,
                    cfg_strength=self.cfg_strength,
                    sway_sampling_coef=-1.0,
                    seed=None,
                )

                # Trim the reference audio from the beginning
                wave = wave[ref_audio_len:]
                mx.eval(wave)

                # Calculate duration directly from wave array
                audio_duration = len(wave) / SAMPLE_RATE

                # Write output file asynchronously
                wave_np = np.array(wave)
                write_future = self.io_executor.submit(sf.write, output_file, wave_np, SAMPLE_RATE)

            except Exception as e:
                # Fallback to original method if custom generation fails
                print(f"Optimized generation failed, falling back to original method: {e}")
                print("Using original F5-TTS generate function (will reload audio file)")
                self.model(
                    generation_text=text,
                    ref_audio_path=voice_file,
                    ref_audio_text=ref_text,
                    output_path=output_file
                )
                audio_duration = None
                write_future = None

            generation_time = time.time() - start_time

            # Ensure the file write completed before returning
            if write_future is not None:
                write_future.result()

            if audio_duration is not None:
                rtf = generation_time / audio_duration if audio_duration > 0 else 0
                print(f"Generated: {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}x)")
                timing_info = {
                    'audio_duration': audio_duration,
                    'generation_time': generation_time,
                    'server_rtf': rtf
                }
                return True, output_file, transcribed_text, timing_info
            elif os.path.exists(output_file):
                # Fallback path -- read duration from file
                try:
                    info = sf.info(output_file)
                    rtf = generation_time / info.duration if info.duration > 0 else 0
                    print(f"Generated: {info.duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}x)")
                    timing_info = {
                        'audio_duration': info.duration,
                        'generation_time': generation_time,
                        'server_rtf': rtf
                    }
                    return True, output_file, transcribed_text, timing_info
                except Exception:
                    print(f"Generated: {output_file} in {generation_time:.2f}s")
                    timing_info = {'generation_time': generation_time, 'server_rtf': None}
                    return True, output_file, transcribed_text, timing_info
            else:
                return False, "Output file not created", None, None

        except Exception as e:
            print(f"Error generating speech: {e}")
            return False, str(e), None, None

    def generate_speech_batch(self, batch_requests):
        """Generate speech for multiple requests in batch with ordered processing"""
        try:
            print(f"Processing batch of {len(batch_requests)} requests...")
            results = []

            # Process all requests in order (important for playback sequence)
            for i, request in enumerate(batch_requests):
                print(f"Batch item {i+1}/{len(batch_requests)}: {request.get('text', '')[:30]}...")

                success, result, transcribed, timing = self.generate_speech(
                    text=request.get('text', ''),
                    voice_file=request.get('voice_file', ''),
                    output_file=request.get('output_file', f'batch_output_{i+1}.wav'),
                    ref_text=request.get('ref_text', None),
                    save_ref_text=request.get('save_ref_text', None)
                )

                results.append({
                    'index': i,
                    'success': success,
                    'result': result,
                    'output_file': request.get('output_file', f'batch_output_{i+1}.wav'),
                    'transcribed': transcribed,
                    'timing': timing
                })

                if not success:
                    print(f"Batch item {i+1} failed: {result}")
                else:
                    print(f"Batch item {i+1} completed: {result}")

            successful = sum(1 for r in results if r['success'])
            print(f"Batch complete: {successful}/{len(batch_requests)} items successful")

            return results
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return [{'success': False, 'result': str(e), 'index': i} for i in range(len(batch_requests))]

    def prewarm_voice_cache(self, batch_requests):
        """Pre-warm the audio and pinyin caches for a batch of requests.

        Args:
            batch_requests: list of request dicts with 'voice_file' and optionally 'ref_text'.

        Returns:
            dict mapping voice_file -> cached expanded audio tensor.
        """
        # Deduplicate by voice file
        voice_to_ref = {}
        for req in batch_requests:
            vf = req.get('voice_file', '')
            if vf and vf not in voice_to_ref:
                voice_to_ref[vf] = req.get('ref_text')

        print(f"Pre-warming cache for {len(voice_to_ref)} unique voice(s)...")

        prewarmed = {}
        for voice_file, ref_text in voice_to_ref.items():
            if not os.path.exists(voice_file):
                print(f"Voice file not found: {voice_file}")
                continue

            # Check if already cached
            cached = self.get_cached_audio(voice_file)
            if cached is not None:
                print(f"Already cached: {os.path.basename(voice_file)}")
                prewarmed[voice_file] = cached
            else:
                try:
                    audio, sr = sf.read(voice_file)
                    if sr != 24000:
                        print(f"Voice {voice_file} has wrong sample rate: {sr}")
                        continue

                    audio = mx.array(audio)
                    TARGET_RMS = 0.1
                    rms = mx.sqrt(mx.mean(mx.square(audio)))
                    if rms < TARGET_RMS:
                        audio = audio * TARGET_RMS / rms

                    audio_expanded = mx.expand_dims(audio, axis=0)
                    self.cache_audio(voice_file, audio_expanded)
                    prewarmed[voice_file] = audio_expanded
                    print(f"Cached: {os.path.basename(voice_file)}")
                except Exception as e:
                    print(f"Failed to cache {voice_file}: {e}")

            # Pre-compute pinyin for ref_text if available
            if ref_text:
                self.get_ref_text_pinyin(ref_text)

        print(f"Voice cache pre-warming complete")
        return prewarmed

    def generate_speech_batch_streaming(self, batch_requests, client_socket):
        """Generate speech for batch with streaming results - sends each result immediately"""
        try:
            total = len(batch_requests)
            print(f"Streaming batch of {total} requests...")

            # Pre-warm voice and pinyin caches for all voices in the batch
            self.prewarm_voice_cache(batch_requests)

            successful = 0

            # Process each request and stream result immediately
            for i, request in enumerate(batch_requests):
                print(f"Streaming item {i+1}/{total}: {request.get('text', '')[:30]}...")

                success, result, transcribed, timing = self.generate_speech(
                    text=request.get('text', ''),
                    voice_file=request.get('voice_file', ''),
                    output_file=request.get('output_file', f'batch_output_{i+1}.wav'),
                    ref_text=request.get('ref_text', None),
                    save_ref_text=request.get('save_ref_text', None)
                )

                # Build result for this chunk
                chunk_result = {
                    'type': 'chunk',
                    'index': i,
                    'success': success,
                    'result': result,
                    'output_file': request.get('output_file', f'batch_output_{i+1}.wav'),
                    'transcribed': transcribed,
                    'timing': timing
                }

                if success:
                    successful += 1
                    print(f"Streamed item {i+1}: {result}")
                else:
                    print(f"Streamed item {i+1} failed: {result}")

                # Send this result immediately (newline-delimited JSON)
                try:
                    response_line = json.dumps(chunk_result) + '\n'
                    client_socket.sendall(response_line.encode('utf-8'))
                except Exception as e:
                    print(f"Failed to send streaming result: {e}")
                    break

            # Send final completion message
            completion = {
                'type': 'complete',
                'success': True,
                'total_items': total,
                'successful_items': successful
            }
            try:
                client_socket.sendall((json.dumps(completion) + '\n').encode('utf-8'))
            except Exception as e:
                print(f"Failed to send completion message: {e}")

            print(f"Streaming batch complete: {successful}/{total} items successful")

        except Exception as e:
            print(f"Error in streaming batch: {e}")
            error_msg = {
                'type': 'error',
                'success': False,
                'error': str(e)
            }
            try:
                client_socket.sendall((json.dumps(error_msg) + '\n').encode('utf-8'))
            except:
                pass

    def _recv_all(self, sock, length):
        """Receive exactly `length` bytes from socket."""
        chunks = []
        received = 0
        while received < length:
            chunk = sock.recv(min(length - received, 65536))
            if not chunk:
                raise ConnectionError("Connection closed before all data received")
            chunks.append(chunk)
            received += len(chunk)
        return b''.join(chunks)

    def _recv_request(self, sock):
        """Receive a length-prefixed JSON request.

        Protocol: 4-byte big-endian length header followed by UTF-8 JSON payload.
        Falls back to newline-delimited read for backwards compatibility.
        """
        # Peek at first 4 bytes to detect protocol
        header = self._recv_all(sock, 4)

        # Try interpreting as a length prefix
        payload_len = struct.unpack('!I', header)[0]

        # Sanity check: valid JSON starts with '{' (0x7b) or '[' (0x5b).
        # A 4-byte length prefix for any reasonable payload will have a first
        # byte that is NOT one of these ASCII characters (payloads < 8MB).
        first_byte = header[0]
        if first_byte in (0x7b, 0x5b):
            # This looks like raw JSON (legacy client), not a length prefix.
            # Read the rest using the old approach: accumulate until valid JSON.
            data = header
            while True:
                try:
                    return json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    pass
                chunk = sock.recv(65536)
                if not chunk:
                    # No more data -- try one last parse
                    return json.loads(data.decode('utf-8'))
                data += chunk
        else:
            # Length-prefixed protocol
            if payload_len > 100 * 1024 * 1024:  # 100MB sanity limit
                raise ValueError(f"Payload length {payload_len} exceeds 100MB limit")
            payload = self._recv_all(sock, payload_len)
            return json.loads(payload.decode('utf-8'))

    def handle_client(self, client_socket, addr):
        """Handle individual client requests"""
        try:
            print(f"Client connected: {addr}")

            # Update activity timestamp
            self.update_activity()

            # Receive request using length-prefixed protocol
            try:
                request = self._recv_request(client_socket)
                print(f"Parsed request: {json.dumps(request, indent=2)}")
            except (json.JSONDecodeError, ConnectionError, ValueError) as e:
                print(f"Failed to receive request from {addr}: {e}")
                return

            # Check if this is a batch request
            if 'batch' in request:
                batch_items = request['batch']
                stream_mode = request.get('stream', False)

                if stream_mode:
                    print(f"Streaming batch request with {len(batch_items)} items")
                    self.generate_speech_batch_streaming(batch_items, client_socket)
                    return
                else:
                    print(f"Batch request with {len(batch_items)} items")
                    results = self.generate_speech_batch(batch_items)

                    # Send batch response
                    response = {
                        'success': True,
                        'batch_results': results,
                        'total_items': len(results),
                        'successful_items': sum(1 for r in results if r['success'])
                    }
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
                    return

            # Handle single request (existing logic)
            text = request.get('text', '')
            voice_file = request.get('voice_file', '')
            output_file = request.get('output_file', 'daemon_output.wav')
            ref_text = request.get('ref_text', None)
            play_audio = request.get('play', False)
            save_ref_text = request.get('save_ref_text', None)

            # Generate speech
            success, result, transcribed_text, timing_info = self.generate_speech(text, voice_file, output_file, ref_text, save_ref_text)

            # Play audio if requested
            if success and play_audio:
                try:
                    time.sleep(0.1)
                    subprocess.run(["sox", result, "-d"], check=True, capture_output=True)
                    print("Audio played")
                except Exception as e:
                    print(f"Could not play audio: {e}")

            # Send response
            response = {
                'success': success,
                'result': result,
                'output_file': output_file if success else None,
                'timing': timing_info if success else None
            }

            client_socket.sendall(json.dumps(response).encode('utf-8'))

        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            error_response = {
                'success': False,
                'result': str(e),
                'output_file': None
            }
            try:
                client_socket.sendall(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()
            print(f"Client disconnected: {addr}")

    def start(self):
        """Start the daemon server"""
        if not self.load_model():
            return False

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.settimeout(1.0)  # 1 second timeout for accept()
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
            self.running = True

            print(f"F5-TTS Daemon started on {self.host}:{self.port}")
            if self.idle_timeout > 0:
                print(f"Will auto-shutdown after {self.idle_timeout//60} minutes of inactivity")
            else:
                print("Idle timeout disabled")
            print("Using Apple Silicon MLX acceleration")
            print(f"Voice audio processing cache: {self.cache_dir}")
            print("Use Ctrl+C to stop")

            # Initialize activity timestamp now that daemon is ready
            self.update_activity()

            # Start idle monitoring thread
            idle_thread = threading.Thread(target=self.idle_monitor)
            idle_thread.daemon = True
            idle_thread.start()

            while self.running:
                try:
                    client_sock, addr = self.sock.accept()
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_sock, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.timeout:
                    # Timeout allows us to check self.running flag
                    continue
                except socket.error:
                    if self.running:
                        print("Socket error occurred")
                    break

        except KeyboardInterrupt:
            print("\nStopping daemon...")
        except Exception as e:
            print(f"Daemon error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the daemon server"""
        self.running = False
        if self.sock:
            self.sock.close()
        print("Daemon stopped")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="F5-TTS Daemon Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9998, help="Port to bind to")
    parser.add_argument("--idle-timeout", type=int, default=900, help="Auto-shutdown after idle seconds (default: 900 = 15 minutes)")
    parser.add_argument("--quantization", "-q", type=int, choices=[4, 8], help="Model quantization bits (4 or 8) for faster inference and memory savings")

    # Performance tuning parameters
    parser.add_argument("--steps", type=int, default=8, help="Neural ODE sampling steps (default: 8, faster: 4-6)")
    parser.add_argument("--method", choices=["euler", "midpoint", "rk4"], default="rk4", help="Sampling method (default: rk4, faster: euler/midpoint)")
    parser.add_argument("--cfg-strength", type=float, default=2.0, help="Classifier-free guidance strength (default: 2.0, faster: 1.5)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor for duration estimation (default: 1.0)")

    args = parser.parse_args()

    daemon = F5TTSDaemon(
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        quantization_bits=args.quantization,
        steps=args.steps,
        method=args.method,
        cfg_strength=args.cfg_strength,
        speed=args.speed
    )
    daemon.start()

if __name__ == "__main__":
    main()
