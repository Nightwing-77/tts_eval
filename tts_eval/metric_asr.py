"""ASR based metric to measure the adhesiveness of the generated audio to its text transcript."""
import logging
from typing import Optional, Union, List, Callable, Dict
from collections import defaultdict
import io

import numpy as np
from evaluate import load

try:
    import torch
    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
    from transformers import WhisperTokenizer
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from soniox.speech_service import SpeechClient
    SONIOX_AVAILABLE = True
except ImportError:
    SONIOX_AVAILABLE = False


class ASRMetric:
    """Base ASR Metric class with support for multiple ASR backends."""

    def __init__(self,
                 backend: str = "whisper",
                 model_id: Optional[str] = None,
                 api_key: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 attn_implementation: str = "sdpa",
                 device_map: str = "auto",
                 metrics: Union[str, List[str]] = "cer"):
        """
        Initialize ASR Metric.
        
        Args:
            backend: ASR backend to use ("whisper" or "soniox")
            model_id: Model ID for Whisper backend
            api_key: API key for Soniox backend
            torch_dtype: Torch dtype for Whisper
            device: Device for Whisper
            attn_implementation: Attention implementation for Whisper
            device_map: Device map for Whisper
            metrics: Metric(s) to compute (e.g., "cer", "wer")
        """
        self.backend = backend
        logging.info(f"Initializing {backend} ASR backend")

        if backend == "whisper":
            self._init_whisper(model_id, torch_dtype, device, attn_implementation, device_map)
        elif backend == "soniox":
            self._init_soniox(api_key)
        else:
            raise ValueError(f"Unknown ASR backend: {backend}")

        logging.info("Setting up metrics")
        metrics = [metrics] if type(metrics) is str else metrics
        self.metrics = {i: load(i) for i in metrics}

    def _init_whisper(self, model_id, torch_dtype, device, attn_implementation, device_map):
        """Initialize Whisper ASR backend."""
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper backend requires transformers, torch, and evaluate packages")

        if model_id is None:
            model_id = "kotoba-tech/kotoba-whisper-v2.0"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        logging.info("Setting up Whisper pipeline")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            device_map=device_map,
            model_kwargs={"attn_implementation": attn_implementation}
        )

        logging.info("Setting up normalizers")
        basic_normalizer: Callable = BasicTextNormalizer()
        en_normalizer: Callable = EnglishTextNormalizer(WhisperTokenizer.from_pretrained(model_id).english_spelling_normalizer)
        ja_normalizer: Callable = lambda x: basic_normalizer(x).replace(" ", "").replace("。.", "。")
        self.normalizer = defaultdict(lambda: basic_normalizer, {"en": en_normalizer, "ja": ja_normalizer})

    def _init_soniox(self, api_key):
        """Initialize Soniox ASR backend."""
        if not SONIOX_AVAILABLE:
            raise ImportError("Soniox backend requires soniox package")

        if api_key is None:
            raise ValueError("Soniox backend requires api_key parameter")

        logging.info("Setting up Soniox client")
        self.client = SpeechClient(api_key=api_key)
        self.normalizer = defaultdict(lambda: lambda x: x.strip().lower())

    def __call__(self,
                 audio: List[Union[np.ndarray, str]],
                 transcript: str,
                 batch_size: int = 32,
                 language: str = "ja",
                 task: str = "transcribe",
                 sampling_rate: Optional[int] = None,
                 normalize_text: bool = True) -> Dict[str, List[float]]:
        """
        Transcribe audio and compute metrics.
        
        Args:
            audio: List of audio arrays or file paths
            transcript: Reference transcript
            batch_size: Batch size for processing
            language: Language code (e.g., "en", "ja")
            task: Task type ("transcribe" or "translate")
            sampling_rate: Sampling rate for audio arrays
            normalize_text: Whether to normalize text
            
        Returns:
            Dictionary with metric names as keys and lists of scores as values
        """
        if self.backend == "whisper":
            return self._call_whisper(audio, transcript, batch_size, language, task, normalize_text)
        elif self.backend == "soniox":
            return self._call_soniox(audio, transcript, language, sampling_rate, normalize_text)

    def _call_whisper(self, audio, transcript, batch_size, language, task, normalize_text):
        """Transcribe using Whisper backend."""
        result = self.pipe(audio, generate_kwargs={"language": language, "task": task}, batch_size=batch_size)
        text = [i["text"] for i in result]

        if normalize_text:
            text = [self.normalizer[language](t) for t in text]
            transcript = self.normalizer[language](transcript)

        result = {}
        for k, metric in self.metrics.items():
            result[k] = [100 * metric.compute(predictions=[t], references=[transcript]) for t in text]
        return result

    def _call_soniox(self, audio, transcript, language, sampling_rate, normalize_text):
        """Transcribe using Soniox backend."""
        text = []
        
        for audio_item in audio:
            if isinstance(audio_item, str):
                # File path
                with open(audio_item, 'rb') as f:
                    audio_data = f.read()
            else:
                # numpy array - convert to WAV bytes
                import soundfile as sf
                buffer = io.BytesIO()
                if sampling_rate is None:
                    sampling_rate = 16000
                sf.write(buffer, audio_item, sampling_rate, format='wav')
                audio_data = buffer.getvalue()

            # Transcribe with Soniox
            request = {
                "audio_data": audio_data,
                "language_code": self._map_language_code(language),
            }
            
            response = self.client.transcribe(request)
            transcribed_text = response.text if hasattr(response, 'text') else str(response)
            text.append(transcribed_text)

        if normalize_text:
            text = [self.normalizer[language](t) for t in text]
            transcript = self.normalizer[language](transcript)

        result = {}
        for k, metric in self.metrics.items():
            result[k] = [100 * metric.compute(predictions=[t], references=[transcript]) for t in text]
        return result

    @staticmethod
    def _map_language_code(lang: str) -> str:
        """Map language codes to Soniox format."""
        # Soniox uses standard language codes
        mapping = {
            "ja": "ja",
            "en": "en",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "zh": "zh",
        }
        return mapping.get(lang, lang)
