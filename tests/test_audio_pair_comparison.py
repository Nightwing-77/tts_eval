"""Tests for audio pair comparison functionality."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from tts_eval import AudioPair, AudioPairComparator, PairComparisonResult


class TestAudioPair:
    """Test AudioPair dataclass."""

    def test_audio_pair_creation_with_paths(self):
        """Test creating AudioPair with file paths."""
        pair = AudioPair(
            input_audio="input.wav",
            output_audio="output.wav",
            reference_text="test text",
            language="en"
        )
        assert pair.input_audio == "input.wav"
        assert pair.output_audio == "output.wav"
        assert pair.reference_text == "test text"
        assert pair.language == "en"
        assert pair.sampling_rate is None

    def test_audio_pair_creation_with_arrays(self):
        """Test creating AudioPair with numpy arrays."""
        input_array = np.random.randn(16000)
        output_array = np.random.randn(16000)
        
        pair = AudioPair(
            input_audio=input_array,
            output_audio=output_array,
            reference_text="test",
            language="ja",
            sampling_rate=16000
        )
        assert isinstance(pair.input_audio, np.ndarray)
        assert isinstance(pair.output_audio, np.ndarray)
        assert pair.sampling_rate == 16000


class TestPairComparisonResult:
    """Test PairComparisonResult dataclass."""

    def test_result_creation(self):
        """Test creating PairComparisonResult."""
        result = PairComparisonResult(
            input_transcription="test input",
            output_transcription="test output",
            metrics={"cer": {"input": 5.0, "output": 3.0}},
            speaker_similarity=0.95,
            language="en",
            reference_text="reference"
        )
        assert result.input_transcription == "test input"
        assert result.output_transcription == "test output"
        assert result.speaker_similarity == 0.95

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = PairComparisonResult(
            input_transcription="input",
            output_transcription="output",
            metrics={"cer": {"input": 5.0, "output": 3.0}},
            speaker_similarity=0.95,
            language="en",
            reference_text="ref"
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["input_transcription"] == "input"
        assert result_dict["output_transcription"] == "output"

    def test_result_str_formatting(self):
        """Test string representation of result."""
        result = PairComparisonResult(
            input_transcription="input",
            output_transcription="output",
            metrics={"cer": {"input": 5.0, "output": 3.0}},
            speaker_similarity=0.95,
            language="en",
            reference_text="ref"
        )
        result_str = str(result)
        assert "AUDIO PAIR COMPARISON RESULTS" in result_str
        assert "input" in result_str
        assert "output" in result_str


class TestAudioPairComparator:
    """Test AudioPairComparator class."""

    @patch('tts_eval.audio_pair_comparison.SpeechClient')
    @patch('tts_eval.audio_pair_comparison.load')
    def test_comparator_initialization(self, mock_load, mock_client_class):
        """Test comparator initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_metric = MagicMock()
        mock_load.return_value = mock_metric

        with patch('tts_eval.audio_pair_comparison.SONIOX_AVAILABLE', True):
            comparator = AudioPairComparator(
                soniox_api_key="test_key",
                metrics=["cer", "wer"]
            )
            assert comparator.asr is not None
            assert comparator.speaker_similarity is not None
            assert comparator.metrics == ["cer", "wer"]

    @patch('tts_eval.audio_pair_comparison.SpeakerEmbeddingSimilarity')
    @patch('tts_eval.audio_pair_comparison.ASRMetric')
    def test_compare_audio_pair(self, mock_asr_class, mock_speaker_class):
        """Test comparing a single audio pair."""
        # Setup mocks
        mock_asr = MagicMock()
        mock_asr.transcribe.side_effect = ["input transcription", "output transcription"]
        mock_asr.compute_metrics.side_effect = [
            {"cer": 5.0},
            {"cer": 3.0},
            {"wer": 10.0},
            {"wer": 6.0}
        ]
        mock_asr_class.return_value = mock_asr
        
        mock_speaker = MagicMock()
        mock_speaker.return_value = [0.95]
        mock_speaker_class.return_value = mock_speaker
        
        with patch('tts_eval.audio_pair_comparison.SONIOX_AVAILABLE', True):
            comparator = AudioPairComparator(
                soniox_api_key="test_key",
                metrics=["cer", "wer"]
            )
            
            pair = AudioPair(
                input_audio="input.wav",
                output_audio="output.wav",
                reference_text="reference",
                language="en"
            )
            
            result = comparator.compare(pair)
            
            assert result.input_transcription == "input transcription"
            assert result.output_transcription == "output transcription"
            assert result.speaker_similarity == 0.95
            assert result.language == "en"

    @patch('tts_eval.audio_pair_comparison.SpeakerEmbeddingSimilarity')
    @patch('tts_eval.audio_pair_comparison.ASRMetric')
    def test_batch_compare(self, mock_asr_class, mock_speaker_class):
        """Test batch comparison of multiple pairs."""
        # Setup mocks
        mock_asr = MagicMock()
        mock_asr.transcribe.side_effect = [
            "input1", "output1", "input2", "output2"
        ]
        mock_asr.compute_metrics.side_effect = [
            {"cer": 5.0}, {"cer": 3.0}, {"wer": 10.0}, {"wer": 6.0},
            {"cer": 6.0}, {"cer": 4.0}, {"wer": 12.0}, {"wer": 8.0},
        ]
        mock_asr_class.return_value = mock_asr
        
        mock_speaker = MagicMock()
        mock_speaker.return_value = [0.95]
        mock_speaker_class.return_value = mock_speaker
        
        with patch('tts_eval.audio_pair_comparison.SONIOX_AVAILABLE', True):
            comparator = AudioPairComparator(
                soniox_api_key="test_key",
                metrics=["cer", "wer"]
            )
            
            pairs = [
                AudioPair("input1.wav", "output1.wav", "ref1", "en"),
                AudioPair("input2.wav", "output2.wav", "ref2", "en"),
            ]
            
            results = comparator.batch_compare(pairs)
            
            assert len(results) == 2
            assert all(isinstance(r, PairComparisonResult) for r in results)

    def test_comparator_missing_api_key(self):
        """Test error when API key is missing."""
        with patch('tts_eval.audio_pair_comparison.SONIOX_AVAILABLE', False):
            with pytest.raises(ImportError):
                AudioPairComparator(soniox_api_key="test_key")


class TestAudioPairComparatorMetrics:
    """Test metric computation in comparator."""

    @patch('tts_eval.audio_pair_comparison.SpeakerEmbeddingSimilarity')
    @patch('tts_eval.audio_pair_comparison.ASRMetric')
    def test_metrics_single_metric(self, mock_asr_class, mock_speaker_class):
        """Test with single metric (CER only)."""
        mock_asr = MagicMock()
        mock_asr.transcribe.side_effect = ["input", "output"]
        mock_asr.compute_metrics.side_effect = [
            {"cer": 5.0},
            {"cer": 3.0},
        ]
        mock_asr_class.return_value = mock_asr
        
        mock_speaker = MagicMock()
        mock_speaker.return_value = [0.9]
        mock_speaker_class.return_value = mock_speaker
        
        with patch('tts_eval.audio_pair_comparison.SONIOX_AVAILABLE', True):
            comparator = AudioPairComparator(
                soniox_api_key="test_key",
                metrics="cer"
            )
            
            pair = AudioPair("input.wav", "output.wav", "ref", "en")
            result = comparator.compare(pair)
            
            assert "cer" in result.metrics

    @patch('tts_eval.audio_pair_comparison.SpeakerEmbeddingSimilarity')
    @patch('tts_eval.audio_pair_comparison.ASRMetric')
    def test_metrics_multiple_metrics(self, mock_asr_class, mock_speaker_class):
        """Test with multiple metrics (CER and WER)."""
        mock_asr = MagicMock()
        mock_asr.transcribe.side_effect = ["input", "output"]
        mock_asr.compute_metrics.side_effect = [
            {"cer": 5.0},
            {"cer": 3.0},
            {"wer": 10.0},
            {"wer": 6.0},
        ]
        mock_asr_class.return_value = mock_asr
        
        mock_speaker = MagicMock()
        mock_speaker.return_value = [0.9]
        mock_speaker_class.return_value = mock_speaker
        
        with patch('tts_eval.audio_pair_comparison.SONIOX_AVAILABLE', True):
            comparator = AudioPairComparator(
                soniox_api_key="test_key",
                metrics=["cer", "wer"]
            )
            
            pair = AudioPair("input.wav", "output.wav", "ref", "en")
            result = comparator.compare(pair)
            
            assert "cer" in result.metrics
            assert "wer" in result.metrics
