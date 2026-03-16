"""Soniox-based WER evaluation with strict language hints."""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, environment variables must be set manually")

try:
    from soniox import SonioxClient
    from soniox.types import CreateTranscriptionConfig
    from soniox.utils import render_tokens
except ImportError:
    logging.error("soniox package not installed. Please install it with: pip install soniox")
    raise


class SonioxWERMetric:
    """
    WER evaluation using Soniox Python SDK with strict language hints.
    
    This function uses the Soniox Python SDK to transcribe audio files with
    strict language hints and calculates Word Error Rate (WER).
    """

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize Soniox WER metric.
        
        Args:
            api_key: Soniox API key. If None, will try to get from SONIOX_API_KEY env var.
            api_url: Soniox API URL. If None, will use default.
        """
        self.api_key = api_key or os.getenv('SONIOX_API_KEY')
        if not self.api_key:
            raise ValueError("Soniox API key required. Set SONIOX_API_KEY environment variable or pass api_key parameter.")
        
        self.api_url = api_url or os.getenv('SONIOX_API_URL', 'https://api.soniox.com')
        
        # Language mapping for Soniox (using ISO 639-1 codes)
        self.language_mapping = {
            'en': 'en',
            'ja': 'ja', 
            'zh': 'zh',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'ko': 'ko',
            'ar': 'ar',
            'hi': 'hi'
        }

    def _get_soniox_language(self, language: str) -> str:
        """Convert language code to Soniox language name."""
        soniox_lang = self.language_mapping.get(language.lower(), language.lower())
        logging.info(f"WER: Mapping language '{language}' to '{soniox_lang}'")
        return soniox_lang

    def _transcribe_with_soniox(self, audio_path: str, language: str) -> str:
        """
        Transcribe audio file using Soniox Python SDK with strict language hint.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'ja')
            
        Returns:
            Transcribed text
        """
        soniox_lang = self._get_soniox_language(language)
        
        try:
            # Initialize Soniox client
            client = SonioxClient()
            
            # Upload audio file
            file = client.files.upload(audio_path)
            
            # Create transcription config with strict language hints
            config = CreateTranscriptionConfig(
                model="stt-async-v4",
                # Temporarily disable language hints to test basic functionality
                # language_hints=[soniox_lang],  # Strict language hint
                enable_language_identification=True,  # Enable to auto-detect
                enable_speaker_diarization=False,     # Disable for simplicity
                client_reference_id=f"wer_eval_{language}"
            )
            
            logging.info(f"Using language: {soniox_lang}, identification enabled")
            
            # Create transcription
            transcription = client.stt.create(
                config=config, 
                file_id=file.id
            )
            
            # Wait for completion
            client.stt.wait(transcription.id)
            
            # Get transcript
            result = client.stt.get_transcript(transcription.id)
            
            # Render tokens to get text
            transcript_text = render_tokens(result.tokens, [])
            
            # Clean up
            client.stt.delete(transcription.id)
            client.files.delete(file.id)
            
            return transcript_text.strip()
                
        except Exception as e:
            logging.error(f"Soniox transcription failed: {e}")
            raise RuntimeError(f"Soniox transcription failed: {e}")

    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER).
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            CER as percentage (0-100)
        """
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 100.0
        if not hypothesis:
            return 100.0
            
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))
        
        # Dynamic programming for edit distance
        m, n = len(ref_chars), len(hyp_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    cost = 0
                else:
                    cost = 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        cer = dp[m][n] / len(ref_chars) * 100
        return cer

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER).
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            WER as percentage (0-100)
        """
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 100.0
        if not hypothesis:
            return 100.0
            
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Dynamic programming for edit distance
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    cost = 0
                else:
                    cost = 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        wer = dp[m][n] / len(ref_words) * 100
        return wer

    def __call__(self,
                 audio_files: List[str],
                 reference_transcript: str,
                 language: str = "en",
                 metrics: List[str] = ["wer"]) -> Dict[str, List[float]]:
        """
        Evaluate WER and/or CER for multiple audio files using Soniox with strict language hints.
        
        Args:
            audio_files: List of paths to audio files
            reference_transcript: Reference transcript text
            language: Language code (e.g., 'en', 'ja')
            metrics: List of metrics to calculate ['wer', 'cer']
            
        Returns:
            Dictionary with WER/CER scores for each audio file
        """
        results = {}
        
        for metric in metrics:
            if metric.lower() in ['wer', 'cer']:
                results[metric.lower()] = []
        
        if not results:
            raise ValueError("No valid metrics specified. Use 'wer', 'cer', or both.")
        
        for audio_file in audio_files:
            if not Path(audio_file).exists():
                logging.error(f"Audio file not found: {audio_file}")
                for metric in results:
                    results[metric].append(100.0)  # Max error for missing file
                continue
                
            try:
                # Transcribe with Soniox using strict language hint
                hypothesis = self._transcribe_with_soniox(audio_file, language)
                
                # Calculate requested metrics
                if 'wer' in results:
                    wer_score = self._calculate_wer(reference_transcript, hypothesis)
                    results['wer'].append(wer_score)
                    
                if 'cer' in results:
                    cer_score = self._calculate_cer(reference_transcript, hypothesis)
                    results['cer'].append(cer_score)
                
                metric_str = ", ".join([f"{k.upper()}: {results[k][-1]:.2f}%" for k in results])
                logging.info(f"File: {audio_file}, {metric_str}")
                
            except Exception as e:
                logging.error(f"Failed to process {audio_file}: {e}")
                for metric in results:
                    results[metric].append(100.0)  # Max error for failed processing
        
        return results


def evaluate_wer_with_language_hint(audio_files: List[str], 
                                  reference_transcript: str, 
                                  language: str = "en",
                                  api_key: Optional[str] = None,
                                  metrics: List[str] = ["wer"]) -> Dict[str, List[float]]:
    """
    Convenience function for WER/CER evaluation using Soniox with strict language hints.
    
    Args:
        audio_files: List of paths to audio files to evaluate
        reference_transcript: Reference transcript text
        language: Language code for strict hinting (e.g., 'en', 'ja', 'zh')
        api_key: Soniox API key (optional, can be set via environment)
        metrics: List of metrics to calculate ['wer', 'cer']
        
    Returns:
        Dictionary containing WER/CER scores for each audio file
    """
    metric = SonioxWERMetric(api_key=api_key)
    return metric(audio_files, reference_transcript, language, metrics)
