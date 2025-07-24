import numpy as np
import noisereduce as nr
from collections import deque
import time
import logging
from typing import AsyncIterable
from livekit import rtc

logger = logging.getLogger("noise-filter")


class NoiseFilter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

        # Noise estimation
        self.noise_sample_buffer = deque(maxlen=24000)  # 1 second at 24kHz
        self.noise_sample_collected = False
        self.noise_update_interval = 5.0  # seconds
        self.last_noise_update = 0

        # Frame buffering
        self.frame_buffer = deque(maxlen=3)  # Keep last 3 frames for overlap
        self.overlap_samples = 512  # Overlap between frames

        # Processing state
        self.processing_enabled = True
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

        # Performance metrics
        self.processing_times = deque(maxlen=100)
        self.avg_processing_time = 0

    def _update_noise_sample(self, audio_data: np.ndarray, current_time: float) -> None:
        """Update noise sample if enough time has passed"""
        if current_time - self.last_noise_update >= self.noise_update_interval:
            # Only update if audio level is low (likely noise)
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 0.1:  # Threshold for noise
                self.noise_sample_buffer.extend(audio_data)
                self.last_noise_update = current_time
                logger.info("Noise sample updated")

    def _process_frame(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """Process a single audio frame with noise reduction"""
        if not self.enabled:
            return frame

        start_time = time.time()

        try:
            # Convert to numpy array and normalize
            audio_data = np.frombuffer(
                frame.data, dtype=np.int16).astype(np.float32)
            if len(audio_data) == 0:
                return frame

            audio_data = audio_data / 32768.0

            # Skip processing if audio level is very low
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 0.01:  # Very low level, likely silence
                return frame

            # Apply noise reduction
            noise_sample = np.array(self.noise_sample_buffer)
            reduced_audio = nr.reduce_noise(
                y=audio_data,
                sr=frame.sample_rate,
                y_noise=noise_sample,
                n_fft=2048,
                hop_length=512,
                n_jobs=-1,
                stationary=False,
                prop_decrease=0.8  # Less aggressive noise reduction
            )

            # Handle any invalid values
            reduced_audio = np.nan_to_num(
                reduced_audio, nan=0.0, posinf=1.0, neginf=-1.0)
            reduced_audio = np.clip(reduced_audio, -1.0, 1.0)

            # Convert back to int16
            processed_data = (
                reduced_audio * 32768.0).astype(np.int16).tobytes()

            # Update processing time metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.avg_processing_time = np.mean(self.processing_times)

            return rtc.AudioFrame(
                data=processed_data,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=frame.samples_per_channel
            )

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                logger.warning("Too many errors, disabling noise reduction")
                self.processing_enabled = False
            return frame

    async def process_audio_frames(self, frames: AsyncIterable[rtc.AudioFrame]) -> AsyncIterable[rtc.AudioFrame]:
        """Process a stream of audio frames"""
        current_time = time.time()

        async for frame in frames:
            try:
                # Convert to numpy array for noise estimation
                audio_data = np.frombuffer(
                    frame.data, dtype=np.int16).astype(np.float32) / 32768.0

                # Initial noise sample collection
                if not self.noise_sample_collected:
                    self.noise_sample_buffer.extend(audio_data)
                    if len(self.noise_sample_buffer) >= 24000:
                        self.noise_sample_collected = True
                        self.last_noise_update = current_time
                        logger.info("Initial noise sample collected")
                    yield frame
                    continue

                # Update noise sample periodically
                self._update_noise_sample(audio_data, current_time)

                # Process frame if enabled
                if self.processing_enabled:
                    processed_frame = self._process_frame(frame)
                    yield processed_frame
                else:
                    yield frame

            except Exception as e:
                logger.error(f"Error in audio pipeline: {str(e)}")
                yield frame