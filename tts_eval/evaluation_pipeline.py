"""Multilingual TTS Evaluation Pipeline combining ASR and speaker embedding metrics."""

import logging
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import numpy as np

from .metric_asr import ASRMetric
from .metric_speaker_embedding_similarity import SpeakerEmbeddingSimilarity
from .languages import get_language_config


@dataclass
class AudioSample:
    """Audio sample with metadata."""
    audio: Union[np.ndarray, str]  # Audio array or file path
    transcript: str
    reference_speaker_audio: Optional[Union[np.ndarray, str]] = None
    language: str = "en"
    sampling_rate: Optional[int] = None


@dataclass
class EvaluationResult:
    """Result of TTS evaluation for a single sample."""
    sample_id: str
    language: str
    asr_scores: Dict[str, float]  # WER, CER, etc.
    speaker_similarity_scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class TTSEvaluationPipeline:
    """
    Unified pipeline for evaluating TTS output across multiple dimensions:
    - ASR metrics (WER/CER) to measure transcription accuracy
    - Speaker embedding similarity to measure speaker preservation
    """

    def __init__(self,
                 asr_backend: str = "whisper",
                 asr_model_id: Optional[str] = None,
                 soniox_api_key: Optional[str] = None,
                 speaker_embedding_model: str = "pyannote",
                 asr_metrics: Union[str, List[str]] = ["cer", "wer"],
                 speaker_metric: str = "cosine_similarity",
                 device: Optional[str] = None):
        """
        Initialize the TTS evaluation pipeline.

        Args:
            asr_backend: ASR backend ("whisper" or "soniox")
            asr_model_id: Model ID for Whisper backend
            soniox_api_key: API key for Soniox backend
            speaker_embedding_model: Speaker embedding model name
            asr_metrics: Metrics to compute for ASR (e.g., ["cer", "wer"])
            speaker_metric: Speaker similarity metric ("cosine_similarity" or "negative_l2_distance")
            device: Device to use for models
        """
        logging.info(f"Initializing TTSEvaluationPipeline with {asr_backend} ASR and {speaker_embedding_model} speaker embedding")

        # Initialize ASR metric
        asr_kwargs = {
            "backend": asr_backend,
            "metrics": asr_metrics,
        }

        if asr_backend == "whisper" and asr_model_id:
            asr_kwargs["model_id"] = asr_model_id
        elif asr_backend == "soniox" and soniox_api_key:
            asr_kwargs["api_key"] = soniox_api_key

        if device:
            asr_kwargs["device"] = device

        self.asr_metric = ASRMetric(**asr_kwargs)
        self.speaker_metric = SpeakerEmbeddingSimilarity(
            model_id=speaker_embedding_model,
            device=device
        )
        self.speaker_similarity_metric = speaker_metric

    def evaluate_sample(self,
                        sample: AudioSample,
                        sample_id: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate a single audio sample.

        Args:
            sample: AudioSample with audio and reference transcript
            sample_id: Optional ID for the sample

        Returns:
            EvaluationResult with ASR and speaker similarity scores
        """
        if sample_id is None:
            sample_id = f"sample_{hash(str(sample.audio)) % 10000}"

        # Compute ASR metrics
        asr_results = self.asr_metric(
            audio=[sample.audio],
            transcript=sample.transcript,
            language=sample.language,
            sampling_rate=sample.sampling_rate,
            normalize_text=True
        )

        # Extract first result (single sample)
        asr_scores = {metric: scores[0] for metric, scores in asr_results.items()}

        # Compute speaker embedding similarity if reference speaker audio provided
        speaker_similarity_scores = None
        if sample.reference_speaker_audio is not None:
            speaker_results = self.speaker_metric(
                audio_target=[sample.audio],
                audio_reference=sample.reference_speaker_audio,
                sampling_rate_target=sample.sampling_rate,
                metric=self.speaker_similarity_metric
            )
            speaker_similarity_scores = {
                self.speaker_similarity_metric: speaker_results[self.speaker_similarity_metric][0]
            }

        return EvaluationResult(
            sample_id=sample_id,
            language=sample.language,
            asr_scores=asr_scores,
            speaker_similarity_scores=speaker_similarity_scores,
            metadata={"model": "tts", "backend": "soniox"}
        )

    def evaluate_batch(self,
                       samples: List[AudioSample],
                       compute_aggregates: bool = True) -> Dict[str, Any]:
        """
        Evaluate a batch of audio samples.

        Args:
            samples: List of AudioSample objects
            compute_aggregates: Whether to compute aggregate statistics

        Returns:
            Dictionary with results and optional aggregate statistics
        """
        results = []
        for idx, sample in enumerate(samples):
            result = self.evaluate_sample(sample, sample_id=f"sample_{idx}")
            results.append(result)

        output = {
            "results": results,
            "num_samples": len(results),
        }

        if compute_aggregates:
            output["aggregates"] = self._compute_aggregates(results)

        return output

    def _compute_aggregates(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute aggregate statistics from evaluation results."""
        aggregates = {}

        # Group by language
        by_language = {}
        for result in results:
            if result.language not in by_language:
                by_language[result.language] = []
            by_language[result.language].append(result)

        # Compute per-language aggregates
        aggregates["by_language"] = {}
        for lang, lang_results in by_language.items():
            asr_aggregates = self._aggregate_asr_scores([r.asr_scores for r in lang_results])
            speaker_aggregates = None

            if any(r.speaker_similarity_scores for r in lang_results):
                speaker_scores = [r.speaker_similarity_scores for r in lang_results if r.speaker_similarity_scores]
                if speaker_scores:
                    speaker_aggregates = self._aggregate_speaker_scores(speaker_scores)

            aggregates["by_language"][lang] = {
                "num_samples": len(lang_results),
                "asr": asr_aggregates,
                "speaker_similarity": speaker_aggregates,
            }

        # Compute overall aggregates
        all_asr_scores = [r.asr_scores for r in results]
        aggregates["overall"] = {
            "num_samples": len(results),
            "asr": self._aggregate_asr_scores(all_asr_scores),
        }

        if any(r.speaker_similarity_scores for r in results):
            speaker_scores = [r.speaker_similarity_scores for r in results if r.speaker_similarity_scores]
            if speaker_scores:
                aggregates["overall"]["speaker_similarity"] = self._aggregate_speaker_scores(speaker_scores)

        return aggregates

    @staticmethod
    def _aggregate_asr_scores(scores_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate ASR scores (WER, CER, etc.)."""
        if not scores_list:
            return {}

        metrics = scores_list[0].keys()
        aggregates = {}

        for metric in metrics:
            values = [scores[metric] for scores in scores_list if metric in scores]
            if values:
                aggregates[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return aggregates

    @staticmethod
    def _aggregate_speaker_scores(scores_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate speaker similarity scores."""
        if not scores_list:
            return {}

        # Assume single metric (cosine_similarity or negative_l2_distance)
        metric = list(scores_list[0].keys())[0]
        values = [scores[metric] for scores in scores_list if metric in scores]

        if values:
            return {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            }

        return {}
