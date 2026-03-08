"""Tests for ASR metrics with Whisper and Soniox backends."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from tts_eval import ASRMetric


class TestASRMetricWhisper:
    """Test ASRMetric with Whisper backend."""

    @patch('tts_eval.metric_asr.pipeline')
    @patch('tts_eval.metric_asr.load')
    def test_whisper_initialization(self, mock_load, mock_pipeline):
        """Test Whisper backend initialization."""
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe
        mock_metric = MagicMock()
        mock_load.return_value = mock_metric

        metric = ASRMetric(backend="whisper", metrics="cer")
        assert metric.backend == "whisper"
        mock_pipeline.assert_called_once()

    @patch('tts_eval.metric_asr.pipeline')
    @patch('tts_eval.metric_asr.load')
    def test_whisper_transcription(self, mock_load, mock_pipeline):
        """Test Whisper transcription and metric computation."""
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"text": "test output"}]
        mock_pipeline.return_value = mock_pipe

        mock_metric = MagicMock()
        mock_metric.compute.return_value = 0.05
        mock_load.return_value = mock_metric

        metric = ASRMetric(backend="whisper", metrics="cer")
        
        audio = [np.random.randn(16000)]
        transcript = "test output"
        
        # Mock the normalizer
        metric.normalizer = {str: lambda x: x.lower()}
        
        result = metric(audio, transcript=transcript, language="en", normalize_text=False)
        assert "cer" in result
        assert isinstance(result["cer"], list)

    def test_whisper_missing_dependency(self):
        """Test error when Whisper dependencies are missing."""
        with patch('tts_eval.metric_asr.WHISPER_AVAILABLE', False):
            with pytest.raises(ImportError):
                ASRMetric(backend="whisper")


class TestASRMetricSoniox:
    """Test ASRMetric with Soniox backend."""

    def test_soniox_missing_api_key(self):
        """Test Soniox requires API key."""
        with patch('tts_eval.metric_asr.SONIOX_AVAILABLE', True):
            with pytest.raises(ValueError, match="api_key"):
                ASRMetric(backend="soniox")

    @patch('tts_eval.metric_asr.SpeechClient')
    @patch('tts_eval.metric_asr.load')
    def test_soniox_initialization(self, mock_load, mock_client_class):
        """Test Soniox backend initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_metric = MagicMock()
        mock_load.return_value = mock_metric

        with patch('tts_eval.metric_asr.SONIOX_AVAILABLE', True):
            metric = ASRMetric(backend="soniox", api_key="test_key", metrics="cer")
            assert metric.backend == "soniox"
            mock_client_class.assert_called_once_with(api_key="test_key")

    def test_soniox_missing_dependency(self):
        """Test error when Soniox dependencies are missing."""
        with patch('tts_eval.metric_asr.SONIOX_AVAILABLE', False):
            with pytest.raises(ImportError):
                ASRMetric(backend="soniox", api_key="test_key")

    def test_language_code_mapping(self):
        """Test language code mapping for Soniox."""
        # Map common language codes
        assert ASRMetric._map_language_code("en") == "en"
        assert ASRMetric._map_language_code("ja") == "ja"
        assert ASRMetric._map_language_code("es") == "es"
        assert ASRMetric._map_language_code("fr") == "fr"
        assert ASRMetric._map_language_code("de") == "de"
        assert ASRMetric._map_language_code("zh") == "zh"
        
        # Unknown language passes through
        assert ASRMetric._map_language_code("unknown") == "unknown"


class TestASRMetricBackendSelection:
    """Test ASR backend selection and switching."""

    def test_invalid_backend(self):
        """Test error with invalid backend."""
        with pytest.raises(ValueError, match="Unknown ASR backend"):
            ASRMetric(backend="invalid_backend")

    @patch('tts_eval.metric_asr.pipeline')
    @patch('tts_eval.metric_asr.load')
    def test_metrics_parameter_string(self, mock_load, mock_pipeline):
        """Test metrics parameter as string."""
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe
        mock_metric = MagicMock()
        mock_load.return_value = mock_metric

        metric = ASRMetric(backend="whisper", metrics="cer")
        assert "cer" in metric.metrics

    @patch('tts_eval.metric_asr.pipeline')
    @patch('tts_eval.metric_asr.load')
    def test_metrics_parameter_list(self, mock_load, mock_pipeline):
        """Test metrics parameter as list."""
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe
        mock_metric = MagicMock()
        mock_load.return_value = mock_metric

        metric = ASRMetric(backend="whisper", metrics=["cer", "wer"])
        assert "cer" in metric.metrics
        assert "wer" in metric.metrics
