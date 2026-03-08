"""
Example script demonstrating multilingual TTS evaluation using the unified pipeline.

This script shows how to:
1. Initialize the TTSEvaluationPipeline
2. Evaluate single and batch audio samples
3. Use different languages
4. Compare ASR metrics and speaker similarity
5. Generate aggregate statistics
"""

import numpy as np
import logging
from typing import List, Dict
from pathlib import Path

from tts_eval import (
    TTSEvaluationPipeline,
    AudioSample,
    get_supported_languages,
    get_language_config
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_single_sample_evaluation():
    """Example 1: Evaluate a single audio sample."""
    logger.info("=" * 50)
    logger.info("Example 1: Single Sample Evaluation")
    logger.info("=" * 50)

    # Initialize pipeline with Whisper backend
    pipeline = TTSEvaluationPipeline(
        asr_backend="whisper",
        asr_metrics=["cer", "wer"],
        speaker_embedding_model="pyannote"
    )

    # Create a sample (in real usage, this would be actual audio)
    sample = AudioSample(
        audio=np.random.randn(16000),  # 1 second of audio at 16kHz
        transcript="Hello, this is a test of the TTS system.",
        reference_speaker_audio=np.random.randn(16000),
        language="en",
        sampling_rate=16000
    )

    logger.info(f"Evaluating sample in {sample.language}...")
    
    # This would fail without actual audio, so we'll just show the structure
    logger.info(f"Sample transcript: {sample.transcript}")
    logger.info(f"Sample language: {sample.language}")
    logger.info("(Note: Actual evaluation requires valid audio file)")


def example_2_batch_multilingual_evaluation():
    """Example 2: Evaluate multiple samples across different languages."""
    logger.info("=" * 50)
    logger.info("Example 2: Batch Multilingual Evaluation")
    logger.info("=" * 50)

    # Supported languages
    languages = get_supported_languages()
    logger.info(f"Supported languages ({len(languages)}):")
    for code, name in list(languages.items())[:5]:
        logger.info(f"  - {code}: {name}")
    logger.info(f"  ... and {len(languages) - 5} more")


def example_3_language_specific_configuration():
    """Example 3: Work with language-specific configurations."""
    logger.info("=" * 50)
    logger.info("Example 3: Language Configuration")
    logger.info("=" * 50)

    # Get configurations for different languages
    test_languages = ["en", "ja", "es"]

    for lang_code in test_languages:
        config = get_language_config(lang_code)
        logger.info(f"\nLanguage: {config.name} ({config.code})")
        logger.info(f"  Whisper code: {config.whisper_code}")
        logger.info(f"  Soniox code: {config.soniox_code}")
        logger.info(f"  Script type: {config.script_type}")


def example_4_pipeline_configuration_options():
    """Example 4: Different pipeline configuration options."""
    logger.info("=" * 50)
    logger.info("Example 4: Pipeline Configuration Options")
    logger.info("=" * 50)

    logger.info("\nConfiguration 1: Whisper + Pyannote (Local)")
    logger.info("  - Backend: Whisper (offline)")
    logger.info("  - Metrics: CER, WER")
    logger.info("  - Speaker Model: Pyannote")
    logger.info("  - Use case: Development, privacy-focused")

    logger.info("\nConfiguration 2: Soniox + Meta Voice (Cloud)")
    logger.info("  - Backend: Soniox (60+ languages)")
    logger.info("  - Metrics: CER, WER")
    logger.info("  - Speaker Model: Meta Voice")
    logger.info("  - Use case: Production, multilingual at scale")

    logger.info("\nConfiguration 3: Whisper + CLAP (Multimodal)")
    logger.info("  - Backend: Whisper")
    logger.info("  - Metrics: CER, WER")
    logger.info("  - Speaker Model: CLAP General")
    logger.info("  - Use case: Acoustic feature preservation")


def example_5_metrics_explanation():
    """Example 5: Understanding the metrics."""
    logger.info("=" * 50)
    logger.info("Example 5: Metrics Explanation")
    logger.info("=" * 50)

    logger.info("\nASR Metrics (lower is better):")
    logger.info("  - WER (Word Error Rate): Percentage of words that differ")
    logger.info("    Example: 'hello world' vs 'helo world' = 50% WER")
    logger.info("  - CER (Character Error Rate): Percentage of characters that differ")
    logger.info("    Example: 'hello' vs 'hallo' = 20% CER")

    logger.info("\nSpeaker Similarity Metrics:")
    logger.info("  - Cosine Similarity: 0-1 scale (higher is better)")
    logger.info("    0.9+: Excellent speaker match")
    logger.info("    0.8-0.9: Good speaker match")
    logger.info("    0.7-0.8: Fair speaker match")
    logger.info("    <0.7: Poor speaker match")


def example_6_asr_backend_comparison():
    """Example 6: Compare ASR backend characteristics."""
    logger.info("=" * 50)
    logger.info("Example 6: ASR Backend Comparison")
    logger.info("=" * 50)

    logger.info("\nWhisper Backend:")
    logger.info("  Pros:")
    logger.info("    ✓ No API key required")
    logger.info("    ✓ Offline, privacy-focused")
    logger.info("    ✓ Good for development/testing")
    logger.info("  Cons:")
    logger.info("    ✗ Requires GPU for good performance")
    logger.info("    ✗ Slower than cloud APIs")
    logger.info("    ✗ Large model downloads")

    logger.info("\nSoniox Backend:")
    logger.info("  Pros:")
    logger.info("    ✓ 60+ language support")
    logger.info("    ✓ High accuracy")
    logger.info("    ✓ Cloud-based, no local compute")
    logger.info("    ✓ Fast inference")
    logger.info("  Cons:")
    logger.info("    ✗ Requires API key")
    logger.info("    ✗ Cloud-dependent (costs, latency)")
    logger.info("    ✗ Needs internet connection")


def example_7_output_structure():
    """Example 7: Understand the output structure."""
    logger.info("=" * 50)
    logger.info("Example 7: Output Structure")
    logger.info("=" * 50)

    logger.info("\nEvaluationResult Structure:")
    logger.info("""
    {
        'sample_id': 'sample_001',
        'language': 'en',
        'asr_scores': {
            'cer': 5.2,      # Character Error Rate (%)
            'wer': 3.1       # Word Error Rate (%)
        },
        'speaker_similarity_scores': {
            'cosine_similarity': 0.87   # 0-1 scale
        },
        'metadata': {...}
    }
    """)

    logger.info("\nBatch Results Aggregates Structure:")
    logger.info("""
    {
        'results': [...],  # List of EvaluationResult objects
        'num_samples': 100,
        'aggregates': {
            'by_language': {
                'en': {
                    'num_samples': 50,
                    'asr': {
                        'cer': {'mean': 5.2, 'std': 1.3, 'min': 2.1, 'max': 8.9},
                        'wer': {'mean': 3.1, 'std': 0.8, 'min': 1.2, 'max': 5.6}
                    },
                    'speaker_similarity': {...}
                },
                'ja': {...}
            },
            'overall': {
                'num_samples': 100,
                'asr': {...},
                'speaker_similarity': {...}
            }
        }
    }
    """)


def example_8_practical_workflow():
    """Example 8: Practical workflow for TTS evaluation."""
    logger.info("=" * 50)
    logger.info("Example 8: Practical Workflow")
    logger.info("=" * 50)

    logger.info("\nStep-by-step workflow:")
    logger.info("1. Initialize pipeline with desired configuration")
    logger.info("   - Choose ASR backend (Whisper or Soniox)")
    logger.info("   - Select metrics (CER, WER)")
    logger.info("   - Choose speaker embedding model")
    logger.info("")
    logger.info("2. Prepare audio samples")
    logger.info("   - Load generated TTS audio")
    logger.info("   - Load reference transcripts")
    logger.info("   - Optionally load reference speaker audio")
    logger.info("")
    logger.info("3. Create AudioSample objects")
    logger.info("   - Specify language for each sample")
    logger.info("   - Set sampling rate if using audio arrays")
    logger.info("")
    logger.info("4. Run evaluation")
    logger.info("   - Single sample: evaluate_sample()")
    logger.info("   - Batch: evaluate_batch()")
    logger.info("")
    logger.info("5. Analyze results")
    logger.info("   - Review per-sample scores")
    logger.info("   - Check language-specific aggregates")
    logger.info("   - Compare against baselines")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 50)
    logger.info("Multilingual TTS Evaluation Framework - Examples")
    logger.info("=" * 50 + "\n")

    # Run all examples
    example_1_single_sample_evaluation()
    example_2_batch_multilingual_evaluation()
    example_3_language_specific_configuration()
    example_4_pipeline_configuration_options()
    example_5_metrics_explanation()
    example_6_asr_backend_comparison()
    example_7_output_structure()
    example_8_practical_workflow()

    logger.info("\n" + "=" * 50)
    logger.info("Examples Complete")
    logger.info("=" * 50)
    logger.info("\nFor more information, see MULTILINGUAL_FRAMEWORK.md")


if __name__ == "__main__":
    main()
