#!/usr/bin/env python3
"""
F5-TTS Daemon Server
Keeps F5-TTS model loaded in memory with Apple Silicon MLX acceleration
"""
import os
import json
import socket
import subprocess
import threading
import warnings
import time
import hashlib
from pathlib import Path

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

        # Voice audio processing cache
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_dir = Path(__file__).parent.parent / "cache" / "voice_audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_voice_cache_key(self, voice_file):
        """Generate cache key for voice file based on file path and modification time"""
        try:
            stat = os.stat(voice_file)
            file_hash = f"{voice_file}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(file_hash.encode()).hexdigest()
        except OSError:
            return None

    def get_cached_audio(self, voice_file):
        """Get cached processed audio for voice file if available"""
        cache_key = self.get_voice_cache_key(voice_file)
        if not cache_key:
            return None

        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.audio_cache:
                return self.audio_cache[cache_key]

            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.npz"
            if cache_file.exists():
                try:
                    import numpy as np
                    data = np.load(cache_file)
                    audio_array = data['audio']
                    # Load into memory cache as MLX array
                    import mlx.core as mx
                    mx_audio = mx.array(audio_array)
                    self.audio_cache[cache_key] = mx_audio
                    return mx_audio
                except Exception:
                    pass

        return None

    def cache_audio(self, voice_file, processed_audio):
        """Cache processed audio for voice file"""
        cache_key = self.get_voice_cache_key(voice_file)
        if not cache_key or processed_audio is None:
            return

        with self.cache_lock:
            # Store in memory cache
            self.audio_cache[cache_key] = processed_audio

            # Store in disk cache
            cache_file = self.cache_dir / f"{cache_key}.npz"
            try:
                import numpy as np
                # Convert MLX array to numpy for storage
                audio_np = np.array(processed_audio)
                np.savez_compressed(cache_file, audio=audio_np)
            except Exception as e:
                print(f"Failed to cache processed audio: {e}")

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
            from f5_tts_mlx.generate import generate
            from f5_tts_mlx.cfm import F5TTS

            if self.quantization_bits:
                print(f"Loading F5-TTS model with {self.quantization_bits}-bit quantization...")
                print(f"Expected memory savings: ~{40 if self.quantization_bits == 8 else 50}%")
            else:
                print("Loading F5-TTS model (full precision)...")

            # Store both the generate function and the model instance with quantization
            self.model = generate
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

            return True
        except Exception as e:
            print(f"Error loading F5-TTS: {e}")
            return False

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

            # Try to get cached processed audio first
            cache_start = time.time()
            processed_audio = self.get_cached_audio(voice_file)
            cache_time = time.time() - cache_start

            if processed_audio is not None:
                print(f"CACHE HIT: Loaded processed audio in {cache_time*1000:.2f}ms for: {os.path.basename(voice_file)}")
                print(f"Audio shape: {processed_audio.shape}, skipping file I/O and processing")
            else:
                print(f"CACHE MISS: No cached audio found for: {os.path.basename(voice_file)}")
                # Load and process audio file
                load_start = time.time()
                print(f"Loading and processing audio file...")
                import soundfile as sf
                import mlx.core as mx

                # Load reference audio
                audio, sr = sf.read(voice_file)
                if sr != 24000:  # F5-TTS requires 24kHz
                    raise ValueError(f"Reference audio must have a sample rate of 24kHz, got {sr}")
                load_time = time.time() - load_start

                # Convert to MLX array and normalize
                process_start = time.time()
                audio = mx.array(audio)
                TARGET_RMS = 0.1
                rms = mx.sqrt(mx.mean(mx.square(audio)))
                if rms < TARGET_RMS:
                    audio = audio * TARGET_RMS / rms
                process_time = time.time() - process_start

                processed_audio = audio

                # Cache the processed audio
                cache_save_start = time.time()
                self.cache_audio(voice_file, processed_audio)
                cache_save_time = time.time() - cache_save_start

                total_processing = load_time + process_time + cache_save_time
                print(f"CACHED: Load {load_time*1000:.1f}ms + Process {process_time*1000:.1f}ms + Save {cache_save_time*1000:.1f}ms = {total_processing*1000:.1f}ms total")

            transcribed_text = None

            print(f"Generating: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            start_time = time.time()

            # Generate speech using MLX with cached processed audio
            print("Using optimized generation with cached audio")
            try:
                from f5_tts_mlx.generate import SAMPLE_RATE, HOP_LENGTH, FRAMES_PER_SEC, split_sentences
                from f5_tts_mlx.utils import convert_char_to_pinyin
                from f5_tts_mlx.cfm import F5TTS
                import mlx.core as mx
                import soundfile as sf
                import numpy as np

                # Use the loaded model instance from daemon
                f5tts = self.f5tts_instance

                # Process text
                generation_text_processed = convert_char_to_pinyin([ref_text + " " + text])

                # Generate using our cached processed audio with tunable parameters
                wave, _ = f5tts.sample(
                    mx.expand_dims(processed_audio, axis=0),
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
                wave = wave[processed_audio.shape[0]:]
                mx.eval(wave)

                # Save the output
                sf.write(output_file, np.array(wave), SAMPLE_RATE)

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

            generation_time = time.time() - start_time

            if os.path.exists(output_file):
                import soundfile as sf
                try:
                    info = sf.info(output_file)
                    rtf = generation_time / info.duration if info.duration > 0 else 0
                    print(f"Generated: {info.duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}x)")

                    # Return timing information for client
                    timing_info = {
                        'audio_duration': info.duration,
                        'generation_time': generation_time,
                        'server_rtf': rtf
                    }
                    return True, output_file, transcribed_text, timing_info
                except:
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

    def prewarm_voice_cache(self, voice_files):
        """Pre-warm the voice cache for a list of voice files"""
        unique_voices = set(voice_files)
        print(f"Pre-warming cache for {len(unique_voices)} unique voice(s)...")

        import soundfile as sf
        import mlx.core as mx

        for voice_file in unique_voices:
            if not os.path.exists(voice_file):
                print(f"Voice file not found: {voice_file}")
                continue

            # Check if already cached
            if self.get_cached_audio(voice_file) is not None:
                print(f"Already cached: {os.path.basename(voice_file)}")
                continue

            try:
                # Load and process audio
                audio, sr = sf.read(voice_file)
                if sr != 24000:
                    print(f"Voice {voice_file} has wrong sample rate: {sr}")
                    continue

                audio = mx.array(audio)
                TARGET_RMS = 0.1
                rms = mx.sqrt(mx.mean(mx.square(audio)))
                if rms < TARGET_RMS:
                    audio = audio * TARGET_RMS / rms

                self.cache_audio(voice_file, audio)
                print(f"Cached: {os.path.basename(voice_file)}")
            except Exception as e:
                print(f"Failed to cache {voice_file}: {e}")

        print(f"Voice cache pre-warming complete")

    def generate_speech_batch_streaming(self, batch_requests, client_socket):
        """Generate speech for batch with streaming results - sends each result immediately"""
        try:
            total = len(batch_requests)
            print(f"Streaming batch of {total} requests...")

            # Pre-warm voice cache for all voices in the batch
            voice_files = [req.get('voice_file', '') for req in batch_requests]
            self.prewarm_voice_cache(voice_files)

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
                    client_socket.send(response_line.encode('utf-8'))
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
                client_socket.send((json.dumps(completion) + '\n').encode('utf-8'))
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
                client_socket.send((json.dumps(error_msg) + '\n').encode('utf-8'))
            except:
                pass

    def handle_client(self, client_socket, addr):
        """Handle individual client requests"""
        try:
            print(f"Client connected: {addr}")

            # Update activity timestamp
            self.update_activity()

            # Receive request
            data = client_socket.recv(4096).decode('utf-8')
            print(f"Raw request data: {repr(data)}")

            if not data:
                print(f"No data received from {addr}")
                return

            try:
                request = json.loads(data)
                print(f"Parsed request: {json.dumps(request, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error from {addr}: {e}")
                print(f"Raw data was: {repr(data)}")
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
                    client_socket.send(json.dumps(response).encode('utf-8'))
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

            client_socket.send(json.dumps(response).encode('utf-8'))

        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            error_response = {
                'success': False,
                'result': str(e),
                'output_file': None
            }
            try:
                client_socket.send(json.dumps(error_response).encode('utf-8'))
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
