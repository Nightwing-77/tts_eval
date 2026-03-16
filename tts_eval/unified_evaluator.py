"""Unified TTS Evaluator with Soniox WER/CER and Speaker Embedding Similarity."""
import os
import logging
import tempfile
import json
import subprocess
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, environment variables must be set manually")

try:
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoProcessor
    import soundfile as sf
    import librosa
except ImportError as e:
    logging.warning(f"Some dependencies missing: {e}")


class UnifiedTTSEvaluator:
    """
    Unified TTS evaluation class that combines WER/CER evaluation using Soniox
    with speaker embedding similarity in a single interface.
    
    This class takes a loaded TTS model and provides comprehensive evaluation
    with strict language hints for Soniox transcription.
    """

    def __init__(self, 
                 tts_model: Any = None,
                 soniox_api_key: Optional[str] = None,
                 soniox_api_url: Optional[str] = None,
                 speaker_embedding_model: str = "metavoice"):
        """
        Initialize the unified TTS evaluator.
        
        Args:
            tts_model: Loaded TTS model (e.g., from transformers.AutoModelForCausalLM)
            soniox_api_key: Soniox API key. If None, will try to get from SONIOX_API_KEY env var.
            soniox_api_url: Soniox API URL. If None, will use default.
            speaker_embedding_model: Model for speaker embedding similarity
        """
        self.tts_model = tts_model
        self.soniox_api_key = soniox_api_key or os.getenv('SONIOX_API_KEY')
        if not self.soniox_api_key:
            raise ValueError("Soniox API key required. Set SONIOX_API_KEY environment variable or pass soniox_api_key parameter.")
        
        self.soniox_api_url = soniox_api_url or os.getenv('SONIOX_API_URL', 'https://api.soniox.com')
        self.speaker_embedding_model = speaker_embedding_model
        
        # Language mapping for Soniox with strict hints
        self.language_mapping = {
            'en': 'english',
            'ja': 'japanese', 
            'zh': 'chinese',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian',
            'ko': 'korean',
            'ar': 'arabic',
            'hi': 'hindi'
        }
        
        # Initialize speaker embedding model
        self._init_speaker_embedding()

    def _init_speaker_embedding(self):
        """Initialize speaker embedding model."""
        try:
            from .speaker_embedding import speaker_embeddings
            if self.speaker_embedding_model in speaker_embeddings:
                self.speaker_embedder = speaker_embeddings[self.speaker_embedding_model]()
            else:
                raise ValueError(f"Speaker embedding model '{self.speaker_embedding_model}' not available")
        except ImportError:
            logging.warning("Speaker embedding not available, similarity scores will be skipped")
            self.speaker_embedder = None

    def _get_soniox_language(self, language: str) -> str:
        """Convert language code to Soniox language name."""
        return self.language_mapping.get(language.lower(), language.lower())

    def _generate_speech(self, text: str, language: str) -> Optional[str]:
        """
        Generate speech from text using the provided TTS model.
        
        Args:
            text: Input text to synthesize
            language: Language code
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.tts_model:
            logging.error("No TTS model provided")
            return None
            
        try:
            # This is a placeholder - actual implementation depends on the TTS model
            # You'll need to adapt this based on your specific TTS model
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Example implementation (adapt based on your model):
            # inputs = self.tts_processor(text, return_tensors="pt")
            # with torch.no_grad():
            #     speech = self.tts_model.generate_speech(inputs["input_ids"])
            # sf.write(temp_path, speech.numpy(), self.tts_model.sampling_rate)
            
            logging.warning("TTS generation not implemented - please adapt for your specific model")
            return temp_path
            
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
            return None

    def _transcribe_with_soniox(self, audio_path: str, language: str) -> str:
        """
        Transcribe audio file using Soniox console with strict language hint.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'ja')
            
        Returns:
            Transcribed text
        """
        soniox_lang = self._get_soniox_language(language)
        
        # Create temporary config file for Soniox
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            config = {
                "api_key": self.soniox_api_key,
                "api_url": self.soniox_api_url
            }
            json.dump(config, config_file)
            config_path = config_file.name
        
        try:
            # Run Soniox console command with strict language hint
            cmd = [
                'soniox', 'transcribe',
                '--audio', audio_path,
                '--language', soniox_lang,
                '--config', config_path,
                '--output-format', 'json',
                '--hint', f'language:{soniox_lang}',  # Strict language hint
                '--hint', f'model:soniox-{soniox_lang}'  # Model-specific hint
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            response = json.loads(result.stdout)
            
            # Extract transcribed text
            if 'transcript' in response:
                return response['transcript']
            elif 'results' in response and response['results']:
                return response['results'][0].get('transcript', '')
            else:
                logging.warning(f"Unexpected response format from Soniox: {response}")
                return ""
                
        except subprocess.CalledProcessError as e:
            logging.error(f"Soniox transcription failed: {e.stderr}")
            raise RuntimeError(f"Soniox transcription failed: {e.stderr}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Soniox response: {e}")
            raise RuntimeError(f"Failed to parse Soniox response: {e}")
        finally:
            # Clean up config file
            os.unlink(config_path)

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate (WER)."""
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

    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate (CER)."""
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

    def _calculate_speaker_similarity(self, reference_audio: str, generated_audio: str) -> Optional[float]:
        """Calculate speaker embedding similarity."""
        if not self.speaker_embedder:
            return None
            
        try:
            ref_embedding = self.speaker_embedder(reference_audio)
            gen_embedding = self.speaker_embedder(generated_audio)
            
            # Cosine similarity
            similarity = np.dot(ref_embedding, gen_embedding) / (
                np.linalg.norm(ref_embedding) * np.linalg.norm(gen_embedding)
            )
            return float(similarity)
        except Exception as e:
            logging.error(f"Speaker similarity calculation failed: {e}")
            return None

    def evaluate(self, 
                 text: str,
                 language: str = "en",
                 reference_audio: Optional[str] = None,
                 metrics: List[str] = ["wer", "cer"]) -> Dict[str, Union[float, str]]:
        """
        Main evaluation function that takes text input and returns comprehensive metrics.
        
        Args:
            text: Input text to evaluate (e.g., "hello how are you")
            language: Language code with strict hinting (e.g., 'en', 'ja', 'zh')
            reference_audio: Path to reference audio for speaker similarity (optional)
            metrics: List of metrics to calculate ['wer', 'cer', 'similarity']
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # Step 1: Generate speech from text (if TTS model provided)
        generated_audio = None
        if self.tts_model:
            generated_audio = self._generate_speech(text, language)
            if generated_audio:
                results['generated_audio_path'] = generated_audio
        
        # Step 2: Transcribe using Soniox with strict language hints
        if generated_audio and Path(generated_audio).exists():
            try:
                transcription = self._transcribe_with_soniox(generated_audio, language)
                results['transcription'] = transcription
                
                # Step 3: Calculate WER/CER
                if 'wer' in metrics:
                    wer_score = self._calculate_wer(text, transcription)
                    results['wer'] = wer_score
                    
                if 'cer' in metrics:
                    cer_score = self._calculate_cer(text, transcription)
                    results['cer'] = cer_score
                    
            except Exception as e:
                logging.error(f"Transcription/evaluation failed: {e}")
                results['error'] = str(e)
                
                # Set max error scores
                if 'wer' in metrics:
                    results['wer'] = 100.0
                if 'cer' in metrics:
                    results['cer'] = 100.0
        
        # Step 4: Calculate speaker similarity (if reference audio provided)
        if reference_audio and 'similarity' in metrics and generated_audio:
            if Path(reference_audio).exists() and Path(generated_audio).exists():
                similarity = self._calculate_speaker_similarity(reference_audio, generated_audio)
                if similarity is not None:
                    results['speaker_similarity'] = similarity
                else:
                    results['speaker_similarity'] = "N/A"
            else:
                results['speaker_similarity'] = "Audio files not found"
        
        # Add metadata
        results.update({
            'input_text': text,
            'language': language,
            'soniox_language_hint': self._get_soniox_language(language)
        })
        
        return results


def evaluate_tts(text: str, 
                language: str = "en",
                tts_model: Any = None,
                reference_audio: Optional[str] = None,
                soniox_api_key: Optional[str] = None,
                metrics: List[str] = ["wer", "cer","similarity"]) -> Dict[str, Union[float, str]]:
    """
    Convenience function for TTS evaluation with strict Soniox language hints.
    
    Args:
        text: Input text to evaluate (e.g., "hello how are you")
        language: Language code with strict hinting (e.g., 'en', 'ja', 'zh')
        tts_model: Loaded TTS model (e.g., from transformers.AutoModelForCausalLM)
        reference_audio: Path to reference audio for speaker similarity (optional)
        soniox_api_key: Soniox API key (optional, can be set via environment)
        metrics: List of metrics to calculate ['wer', 'cer', 'similarity']
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    evaluator = UnifiedTTSEvaluator(
        tts_model=tts_model,
        soniox_api_key=soniox_api_key
    )
    return evaluator.evaluate(text, language, reference_audio, metrics)
