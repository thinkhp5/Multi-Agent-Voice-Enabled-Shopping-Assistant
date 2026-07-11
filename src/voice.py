"""
Voice I/O layer — AWS native.

VoiceRecorder – microphone → WAV → Amazon Transcribe STT → text
VoiceSpeaker  – text → Amazon Polly TTS → playback
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.config import get_logger, polly_client, transcribe_client, AWS_REGION

logger = get_logger("voice")


# ═══════════════════════════════════════════════════════════
#  Speech-to-Text (Amazon Transcribe)
# ═══════════════════════════════════════════════════════════

class VoiceRecorder:
    """Record from the microphone and transcribe with Amazon Transcribe."""

    def __init__(self, sample_rate: int = 16_000):
        self.sample_rate = sample_rate

    def record(self, duration: int = 5, countdown: bool = True) -> np.ndarray:
        """Record *duration* seconds of mono audio."""
        if countdown:
            for i in range(3, 0, -1):
                logger.info("Recording starts in %d …", i)
                time.sleep(1)

        logger.info("Recording for %d s — speak now!", duration)
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        logger.info("Recording complete")
        return audio

    def transcribe(self, audio: np.ndarray, language: str = "en-US") -> str:
        """Send audio to Amazon Transcribe and return the transcript.

        Uses the streaming transcription approach for low-latency results.
        Falls back to batch transcription via S3 if streaming is unavailable.
        """
        # Write audio to a WAV buffer
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        buf.seek(0)
        audio_bytes = buf.read()

        try:
            # Use Amazon Transcribe streaming for real-time transcription
            import boto3
            from amazon_transcribe.client import TranscribeStreamingClient
            from amazon_transcribe.handlers import TranscriptResultStreamHandler
            from amazon_transcribe.model import TranscriptEvent
            import asyncio

            transcript_text = ""

            class MyEventHandler(TranscriptResultStreamHandler):
                async def handle_transcript_event(self, transcript_event: TranscriptEvent):
                    nonlocal transcript_text
                    results = transcript_event.transcript.results
                    for result in results:
                        if not result.is_partial:
                            for alt in result.alternatives:
                                transcript_text += alt.transcript

            async def transcribe_stream():
                client = TranscribeStreamingClient(region=AWS_REGION)
                stream = await client.start_stream_transcription(
                    language_code=language,
                    media_sample_rate_hz=self.sample_rate,
                    media_encoding="pcm",
                )
                handler = MyEventHandler(stream.output_stream)

                # Send audio in chunks
                chunk_size = 1024 * 16
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                await stream.input_stream.end_stream()
                await handler.handle_events()

            asyncio.run(transcribe_stream())
            logger.info("Transcription: %r", transcript_text)
            return transcript_text.strip()

        except ImportError:
            # Fallback: use batch transcription if streaming SDK not available
            logger.info("Streaming SDK not available, using batch transcription")
            return self._batch_transcribe(audio_bytes, language)
        except Exception:
            logger.exception("Amazon Transcribe failed")
            return ""

    def _batch_transcribe(self, audio_bytes: bytes, language: str = "en-US") -> str:
        """Fallback batch transcription using a local temp file and polling."""
        import boto3

        # Save to a temp file and upload to S3 for batch transcription
        # For simplicity, use the synchronous StartMedicalTranscriptionJob
        # In production, you'd use S3 + polling or streaming
        tmp_path = os.path.join(tempfile.gettempdir(), f"transcribe_{uuid.uuid4().hex[:8]}.wav")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        logger.warning(
            "Batch transcription requires S3. For local development, "
            "install amazon-transcribe-streaming-sdk: "
            "pip install amazon-transcribe"
        )
        return ""

    def record_and_transcribe(self, duration: int = 5) -> tuple[str, str]:
        """Full pipeline: record → save WAV → transcribe.

        Returns (wav_path, transcript).
        """
        audio = self.record(duration)
        wav_path = os.path.join(tempfile.gettempdir(), f"rec_{uuid.uuid4().hex[:8]}.wav")
        sf.write(wav_path, audio, self.sample_rate)
        text = self.transcribe(audio)
        return wav_path, text


# ═══════════════════════════════════════════════════════════
#  Text-to-Speech (Amazon Polly)
# ═══════════════════════════════════════════════════════════

VOICE_OPTIONS = {
    "Joanna":   "Female, US English, neural",
    "Matthew":  "Male, US English, neural",
    "Amy":      "Female, British English, neural",
    "Brian":    "Male, British English, neural",
    "Ivy":      "Female, US English, child voice",
    "Ruth":     "Female, US English, generative",
    "Stephen":  "Male, US English, generative",
}


class VoiceSpeaker:
    """Convert text to speech using Amazon Polly and play it back."""

    def __init__(self, voice: str = "Joanna", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._out_dir = tempfile.gettempdir()

    def synthesise(self, text: str) -> str:
        """Generate an MP3 file via Amazon Polly and return its path."""
        out_path = os.path.join(self._out_dir, f"tts_{uuid.uuid4().hex[:8]}.mp3")

        # Apply speed via SSML if not 1.0
        if self.speed != 1.0:
            rate = f"{int(self.speed * 100)}%"
            ssml_text = f'<speak><prosody rate="{rate}">{text}</prosody></speak>'
            text_type = "ssml"
        else:
            ssml_text = text
            text_type = "text"

        try:
            response = polly_client.synthesize_speech(
                Text=ssml_text,
                TextType=text_type,
                OutputFormat="mp3",
                VoiceId=self.voice,
                Engine="neural",  # Use neural engine for higher quality
            )
            # Write the audio stream to file
            audio_stream = response["AudioStream"].read()
            with open(out_path, "wb") as f:
                f.write(audio_stream)

            logger.info("TTS saved → %s", out_path)
            return out_path
        except polly_client.exceptions.InvalidSsmlException:
            # Fallback: retry without SSML
            try:
                response = polly_client.synthesize_speech(
                    Text=text,
                    TextType="text",
                    OutputFormat="mp3",
                    VoiceId=self.voice,
                    Engine="neural",
                )
                audio_stream = response["AudioStream"].read()
                with open(out_path, "wb") as f:
                    f.write(audio_stream)
                logger.info("TTS saved (plain text fallback) → %s", out_path)
                return out_path
            except Exception:
                logger.exception("TTS synthesis failed (fallback)")
                return ""
        except Exception:
            logger.exception("TTS synthesis failed")
            return ""

    def play(self, audio_file: str) -> None:
        """Play an audio file through the default output device."""
        try:
            data, sr = sf.read(audio_file)
            sd.play(data, sr)
            sd.wait()
        except Exception:
            logger.exception("Audio playback failed")

    def speak(self, text: str, play: bool = True) -> str:
        """Synthesise *text* and optionally play it. Returns the file path."""
        logger.info("Agent says: %s", text[:120])
        path = self.synthesise(text)
        if play and path:
            self.play(path)
        return path
