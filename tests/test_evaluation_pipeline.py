"""Tests for TTSEvaluationPipeline."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from tts_eval import TTSEvaluationPipeline, AudioSample, EvaluationResult


class TestAudioSample:
    """Test AudioSample dataclass."""

    def test_audio_sample_creation(self):
        """Test basic AudioSample creation."""
        audio = np.random.randn(16000)
        sample = AudioSample(
            audio=audio,
            transcript="Hello world",
            language="en",
            sampling_rate=16000
        )
        assert sample.transcript == "Hello world"
        assert sample.language == "en"
        assert sample.sampling_rate == 16000

    def test_audio_sample_with_reference(self):
        """Test AudioSample with reference speaker audio."""
        audio = np.random.randn(16000)
        ref_audio = np.random.randn(16000)
        sample = AudioSample(
            audio=audio,
            transcript="Test",
            reference_speaker_audio=ref_audio,
            language="ja"
        )
        assert sample.reference_speaker_audio is not None


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Test basic EvaluationResult creation."""
        result = EvaluationResult(
            sample_id="test_001",
            language="en",
            asr_scores={"cer": 5.2, "wer": 3.1},
            speaker_similarity_scores={"cosine_similarity": 0.85}
        )
        assert result.sample_id == "test_001"
        assert result.language == "en"
        assert result.asr_scores["cer"] == 5.2


@pytest.mark.parametrize("asr_backend", ["whisper"])
def test_pipeline_initialization(asr_backend):
    """Test TTSEvaluationPipeline initialization."""
    with patch('tts_eval.evaluation_pipeline.ASRMetric'):
        with patch('tts_eval.evaluation_pipeline.SpeakerEmbeddingSimilarity'):
            pipeline = TTSEvaluationPipeline(
                asr_backend=asr_backend,
                asr_metrics=["cer"]
            )
            assert pipeline is not None


@pytest.mark.parametrize("language", ["en", "ja", "es"])
def test_pipeline_multilingual_support(language):
    """Test pipeline supports multiple languages."""
    with patch('tts_eval.evaluation_pipeline.ASRMetric'):
        with patch('tts_eval.evaluation_pipeline.SpeakerEmbeddingSimilarity'):
            pipeline = TTSEvaluationPipeline(asr_backend="whisper")
            
            sample = AudioSample(
                audio=np.random.randn(16000),
                transcript="Test",
                language=language
            )
            assert sample.language == language


def test_evaluation_result_without_speaker_similarity():
    """Test evaluation result without speaker similarity."""
    result = EvaluationResult(
        sample_id="test_001",
        language="en",
        asr_scores={"cer": 5.2}
    )
    assert result.speaker_similarity_scores is None


def test_evaluation_result_with_metadata():
    """Test evaluation result with metadata."""
    metadata = {"model": "gpt2vec", "version": "1.0"}
    result = EvaluationResult(
        sample_id="test_001",
        language="en",
        asr_scores={"cer": 5.2},
        metadata=metadata
    )
    assert result.metadata == metadata
