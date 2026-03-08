"""Audio pair comparison pipeline for TTS evaluation."""
import logging
from dataclasses import dataclass, asdict
from typing import Union, Optional, Dict, List
import numpy as np

from .metric_asr import ASRMetric
from .metric_speaker_embedding_similarity import SpeakerEmbeddingSimilarity


@dataclass
class AudioPair:
    """Represents an input/output audio pair for comparison."""
    input_audio: Union[str, np.ndarray]
    output_audio: Union[str, np.ndarray]
    reference_text: str
    language: str = "en"
    sampling_rate: Optional[int] = None


@dataclass
class PairComparisonResult:
    """Result of comparing two audio files."""
    # Transcription results
    input_transcription: str
    output_transcription: str
    
    # WER/CER metrics
    metrics: Dict[str, Dict[str, float]]  # {"cer": {"input": 5.2, "output": 3.1}, ...}
    
    # Speaker similarity
    speaker_similarity: float
    
    # Language and reference
    language: str
    reference_text: str
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Pretty print results."""
        output = []
        output.append("=" * 60)
        output.append("AUDIO PAIR COMPARISON RESULTS")
        output.append("=" * 60)
        output.append(f"\nLanguage: {self.language}")
        output.append(f"Reference Text: {self.reference_text}")
        
        output.append("\n--- TRANSCRIPTIONS ---")
        output.append(f"Input Transcription:  {self.input_transcription}")
        output.append(f"Output Transcription: {self.output_transcription}")
        
        output.append("\n--- METRICS (WER/CER) ---")
        for metric_name, scores in self.metrics.items():
            input_score = scores.get("input", 0.0)
            output_score = scores.get("output", 0.0)
            improvement = input_score - output_score
            improvement_sign = "✓" if improvement > 0 else "✗" if improvement < 0 else "="
            output.append(f"{metric_name.upper()}:")
            output.append(f"  Input:  {input_score:.2f}%")
            output.append(f"  Output: {output_score:.2f}%")
            output.append(f"  Change: {improvement_sign} {improvement:+.2f}%")
        
        output.append("\n--- SPEAKER SIMILARITY ---")
        output.append(f"Cosine Similarity: {self.speaker_similarity:.4f}")
        output.append("(1.0 = identical speaker, 0.0 = completely different)")
        
        output.append("\n" + "=" * 60)
        return "\n".join(output)


class AudioPairComparator:
    """Pipeline for comparing input/output audio pairs."""

    def __init__(self, soniox_api_key: str, 
                 metrics: Union[str, List[str]] = ["cer", "wer"],
                 speaker_embedding_model: str = "pyannote"):
        """
        Initialize AudioPairComparator.
        
        Args:
            soniox_api_key: Soniox API key for ASR
            metrics: Metrics to compute (cer, wer, etc.)
            speaker_embedding_model: Speaker embedding model ("pyannote" or other)
        """
        logging.info("Initializing AudioPairComparator")
        
        self.asr = ASRMetric(api_key=soniox_api_key, metrics=metrics)
        self.speaker_similarity = SpeakerEmbeddingSimilarity(model_id=speaker_embedding_model)
        self.metrics = [metrics] if isinstance(metrics, str) else metrics

    def compare(self, audio_pair: AudioPair) -> PairComparisonResult:
        """
        Compare input and output audio.
        
        Args:
            audio_pair: AudioPair with input, output audio and reference text
            
        Returns:
            PairComparisonResult with metrics and speaker similarity
        """
        logging.info(f"Comparing audio pair (language: {audio_pair.language})")
        
        # Transcribe both audio files
        logging.debug("Transcribing input audio...")
        input_transcription = self.asr.transcribe(
            audio_pair.input_audio,
            language=audio_pair.language,
            sampling_rate=audio_pair.sampling_rate
        )
        
        logging.debug("Transcribing output audio...")
        output_transcription = self.asr.transcribe(
            audio_pair.output_audio,
            language=audio_pair.language,
            sampling_rate=audio_pair.sampling_rate
        )
        
        # Compute WER/CER for both
        logging.debug("Computing WER/CER metrics...")
        metrics_results = {}
        for metric_name in self.metrics:
            input_metrics = self.asr.compute_metrics(
                input_transcription,
                audio_pair.reference_text,
                normalize=True
            )
            output_metrics = self.asr.compute_metrics(
                output_transcription,
                audio_pair.reference_text,
                normalize=True
            )
            
            metrics_results[metric_name] = {
                "input": input_metrics.get(metric_name, 0.0),
                "output": output_metrics.get(metric_name, 0.0),
            }
        
        # Compute speaker similarity
        logging.debug("Computing speaker similarity...")
        similarity = self.speaker_similarity(
            [audio_pair.input_audio],
            [audio_pair.output_audio],
            sampling_rate=audio_pair.sampling_rate
        )
        
        # Extract single similarity score
        if isinstance(similarity, list) and len(similarity) > 0:
            speaker_sim = similarity[0]
        elif isinstance(similarity, (int, float)):
            speaker_sim = float(similarity)
        else:
            speaker_sim = 0.0
        
        result = PairComparisonResult(
            input_transcription=input_transcription,
            output_transcription=output_transcription,
            metrics=metrics_results,
            speaker_similarity=speaker_sim,
            language=audio_pair.language,
            reference_text=audio_pair.reference_text,
        )
        
        logging.info("Audio pair comparison completed")
        return result

    def batch_compare(self, audio_pairs: List[AudioPair]) -> List[PairComparisonResult]:
        """
        Compare multiple audio pairs.
        
        Args:
            audio_pairs: List of AudioPair objects
            
        Returns:
            List of PairComparisonResult objects
        """
        logging.info(f"Batch comparing {len(audio_pairs)} audio pairs")
        results = []
        for i, pair in enumerate(audio_pairs, 1):
            logging.info(f"Processing pair {i}/{len(audio_pairs)}")
            result = self.compare(pair)
            results.append(result)
        return results
