"""ASR metric using Soniox for multilingual speech-to-text transcription."""
import logging
import io
from typing import Optional, Union, List, Dict

import numpy as np
from evaluate import load

try:
    from soniox.speech_service import SpeechClient
    SONIOX_AVAILABLE = True
except ImportError:
    SONIOX_AVAILABLE = False


class ASRMetric:
    """Soniox-based ASR Metric for transcribing audio and computing WER/CER metrics."""

    def __init__(self, api_key: str, metrics: Union[str, List[str]] = "cer"):
        """
        Initialize Soniox ASR Metric.
        
        Args:
            api_key: Soniox API key for authentication
            metrics: Metric(s) to compute (e.g., "cer", "wer", or ["cer", "wer"])
        """
        if not SONIOX_AVAILABLE:
            raise ImportError("Soniox backend requires the 'soniox' package. Install with: pip install soniox")

        if not api_key:
            raise ValueError("api_key is required for Soniox ASR metric")

        logging.info("Initializing Soniox ASR client")
        self.client = SpeechClient(api_key=api_key)
        
        logging.info("Setting up metrics")
        metrics = [metrics] if isinstance(metrics, str) else metrics
        self.metrics = {metric: load(metric) for metric in metrics}

    def transcribe(self, audio: Union[np.ndarray, str], language: str = "en", 
                   sampling_rate: Optional[int] = None) -> str:
        """
        Transcribe a single audio file or array.
        
        Args:
            audio: Audio file path or numpy array
            language: Language code (e.g., "en", "ja", "es")
            sampling_rate: Sampling rate for numpy arrays (default: 16000)
            
        Returns:
            Transcribed text
        """
        audio_data = self._prepare_audio(audio, sampling_rate)
        
        request = {
            "audio_data": audio_data,
            "language_code": self._map_language_code(language),
        }
        
        try:
            response = self.client.transcribe(request)
            transcribed_text = response.text if hasattr(response, 'text') else str(response)
            logging.debug(f"Transcribed text: {transcribed_text}")
            return transcribed_text.strip()
        except Exception as e:
            logging.error(f"Soniox transcription error: {e}")
            raise

    def compute_metrics(self, predicted_text: str, reference_text: str, 
                       normalize: bool = True) -> Dict[str, float]:
        """
        Compute WER/CER metrics between predicted and reference text.
        
        Args:
            predicted_text: Predicted transcription
            reference_text: Reference transcription
            normalize: Whether to normalize text before comparison
            
        Returns:
            Dictionary with metric names and scores (0-100)
        """
        pred = predicted_text.strip()
        ref = reference_text.strip()
        
        if normalize:
            pred = pred.lower()
            ref = ref.lower()
        
        results = {}
        for metric_name, metric in self.metrics.items():
            score = metric.compute(predictions=[pred], references=[ref])
            results[metric_name] = score * 100  # Convert to percentage
            
        logging.debug(f"Metrics: {results}")
        return results

    def _prepare_audio(self, audio: Union[np.ndarray, str], 
                      sampling_rate: Optional[int] = None) -> bytes:
        """
        Prepare audio data for Soniox API.
        
        Args:
            audio: Audio file path or numpy array
            sampling_rate: Sampling rate for numpy arrays
            
        Returns:
            Audio data as bytes
        """
        if isinstance(audio, str):
            # File path
            with open(audio, 'rb') as f:
                return f.read()
        else:
            # numpy array - convert to WAV bytes
            try:
                import soundfile as sf
            except ImportError:
                raise ImportError("soundfile is required for numpy array audio. Install with: pip install soundfile")
            
            if sampling_rate is None:
                sampling_rate = 16000
                
            buffer = io.BytesIO()
            sf.write(buffer, audio, sampling_rate, format='wav')
            buffer.seek(0)
            return buffer.read()

    @staticmethod
    def _map_language_code(lang: str) -> str:
        """Map language codes to Soniox format."""
        # Soniox supports ISO 639-1 language codes
        mapping = {
            "en": "en",
            "ja": "ja",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "zh": "zh",
            "ko": "ko",
            "pt": "pt",
            "ru": "ru",
            "ar": "ar",
        }
        return mapping.get(lang, lang)
