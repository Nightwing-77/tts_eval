# Multilingual TTS Evaluation Framework

A comprehensive framework for evaluating Text-to-Speech (TTS) systems across multiple languages using ASR metrics and speaker embedding similarity.

## Features

- **Multilingual ASR Support**: Evaluate TTS output across 60+ languages using Soniox or Whisper
- **Dual ASR Backends**: Choose between Soniox (cloud-based, 60+ languages) or Whisper (local, offline)
- **Speaker Embedding Metrics**: Preserve speaker identity verification across languages
- **Unified Pipeline**: Evaluate ASR accuracy (WER/CER) and speaker similarity in one pipeline
- **Language Configuration**: Pre-configured language support with proper text normalization
- **Aggregate Statistics**: Automatic computation of per-language and overall metrics

## Installation

### Basic Installation (Whisper backend)

```bash
pip install -e .
```

### With Soniox Support

```bash
pip install -e ".[soniox]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Using the Unified Pipeline

```python
from tts_eval import TTSEvaluationPipeline, AudioSample
import numpy as np

# Initialize pipeline with Soniox backend
pipeline = TTSEvaluationPipeline(
    asr_backend="soniox",
    soniox_api_key="your-api-key",
    asr_metrics=["cer", "wer"],
    speaker_embedding_model="pyannote"
)

# Create audio sample
sample = AudioSample(
    audio="path/to/generated_audio.wav",
    transcript="Hello, this is a test.",
    reference_speaker_audio="path/to/reference_speaker.wav",
    language="en",
    sampling_rate=16000
)

# Evaluate
result = pipeline.evaluate_sample(sample, sample_id="sample_001")
print(f"CER: {result.asr_scores['cer']:.2f}%")
print(f"Speaker Similarity: {result.speaker_similarity_scores['cosine_similarity']:.3f}")
```

### Batch Evaluation

```python
# Evaluate multiple samples
samples = [
    AudioSample(
        audio=f"audio_{i}.wav",
        transcript="Sample text",
        language="en"
    )
    for i in range(10)
]

results = pipeline.evaluate_batch(samples, compute_aggregates=True)

print(f"Average CER: {results['aggregates']['overall']['asr']['cer']['mean']:.2f}%")
print(f"Average WER: {results['aggregates']['overall']['asr']['wer']['mean']:.2f}%")
```

## Supported Languages

The framework supports 12+ languages out of the box with proper text normalization:

- **English (en)**: Latin script normalization
- **Japanese (ja)**: Space removal and punctuation normalization
- **Spanish (es)**: Latin script normalization
- **French (fr)**: Latin script normalization
- **German (de)**: Latin script normalization
- **Chinese (zh)**: Space removal for CJK
- **Korean (ko)**: Hangul normalization
- **Portuguese (pt)**: Latin script normalization
- **Italian (it)**: Latin script normalization
- **Russian (ru)**: Cyrillic normalization
- **Arabic (ar)**: Arabic script normalization
- **Hindi (hi)**: Devanagari script normalization

### Adding Custom Languages

```python
from tts_eval import LanguageConfig, LANGUAGE_CONFIGS

# Add custom language
custom_lang = LanguageConfig(
    code="my_lang",
    name="My Language",
    whisper_code="my_lang",
    soniox_code="my_lang",
    script_type="custom"
)

LANGUAGE_CONFIGS["my_lang"] = custom_lang
```

## API Reference

### TTSEvaluationPipeline

Main class for TTS evaluation.

#### Initialization

```python
TTSEvaluationPipeline(
    asr_backend: str = "whisper",           # "whisper" or "soniox"
    asr_model_id: Optional[str] = None,     # Model ID for Whisper
    soniox_api_key: Optional[str] = None,   # API key for Soniox
    speaker_embedding_model: str = "pyannote",
    asr_metrics: Union[str, List[str]] = ["cer", "wer"],
    speaker_metric: str = "cosine_similarity",
    device: Optional[str] = None
)
```

#### Methods

- `evaluate_sample(sample: AudioSample, sample_id: Optional[str]) -> EvaluationResult`
- `evaluate_batch(samples: List[AudioSample], compute_aggregates: bool) -> Dict[str, Any]`

### AudioSample

Dataclass representing an audio sample to evaluate.

```python
@dataclass
class AudioSample:
    audio: Union[np.ndarray, str]           # Audio array or file path
    transcript: str                          # Reference transcript
    reference_speaker_audio: Optional[Union[np.ndarray, str]] = None
    language: str = "en"
    sampling_rate: Optional[int] = None
```

### EvaluationResult

Result of evaluating a single sample.

```python
@dataclass
class EvaluationResult:
    sample_id: str
    language: str
    asr_scores: Dict[str, float]             # WER, CER, etc.
    speaker_similarity_scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
```

## Metrics Explanation

### ASR Metrics

- **WER (Word Error Rate)**: Percentage of words that differ from reference
- **CER (Character Error Rate)**: Percentage of characters that differ from reference

### Speaker Embedding Metrics

- **Cosine Similarity**: Similarity between speaker embeddings (0-1, higher is better)
- **Negative L2 Distance**: Negative distance between embeddings (lower magnitude is better)

## Configuration

### Language-Specific Configuration

```python
from tts_eval import get_language_config

# Get language configuration
config = get_language_config("ja")
print(config.whisper_code)  # "ja"
print(config.soniox_code)   # "ja"
print(config.script_type)   # "cjk"
```

### Custom Text Normalizers

```python
from tts_eval import build_normalizers

# Build normalizers for all languages
normalizers = build_normalizers(whisper_model_id="kotoba-tech/kotoba-whisper-v2.0")

# Use custom normalizer
normalized_text = normalizers["ja"]("テスト　テキスト。")
```

## Backend Comparison

### Whisper Backend

**Pros:**
- No API key required
- Offline, privacy-focused
- Multi-language support built-in
- Good for development/testing

**Cons:**
- Requires GPU for good performance
- Slower than cloud APIs
- Larger model downloads

### Soniox Backend

**Pros:**
- 60+ language support
- High accuracy
- Cloud-based, no local compute needed
- Faster inference

**Cons:**
- Requires API key
- Cloud-dependent (latency, costs)
- Needs internet connection

## Examples

### Multilingual Evaluation

```python
from tts_eval import TTSEvaluationPipeline, AudioSample

pipeline = TTSEvaluationPipeline(
    asr_backend="soniox",
    soniox_api_key="your-key"
)

# Evaluate same TTS across multiple languages
languages = ["en", "ja", "es", "fr"]
samples = []

for lang in languages:
    sample = AudioSample(
        audio=f"tts_output_{lang}.wav",
        transcript=f"transcript_{lang}.txt",
        language=lang
    )
    samples.append(sample)

results = pipeline.evaluate_batch(samples, compute_aggregates=True)

# Results grouped by language
for lang, stats in results['aggregates']['by_language'].items():
    print(f"{lang}: CER={stats['asr']['cer']['mean']:.2f}%")
```

### Speaker Identity Preservation

```python
# Test if TTS preserves speaker identity
sample = AudioSample(
    audio="tts_generated.wav",
    transcript="Hello world",
    reference_speaker_audio="original_speaker.wav",
    language="en"
)

result = pipeline.evaluate_sample(sample)

speaker_sim = result.speaker_similarity_scores['cosine_similarity']
print(f"Speaker similarity: {speaker_sim:.3f}")
print(f"Speaker preserved: {'Yes' if speaker_sim > 0.8 else 'No'}")
```

## Performance Tips

1. **Batch Processing**: Process multiple samples at once for better throughput
2. **GPU Acceleration**: Use CUDA for Whisper backend with `device="cuda"`
3. **Caching**: Cache model weights and speaker embeddings across evaluations
4. **Language Detection**: Pre-detect language to avoid overhead

## Troubleshooting

### ImportError: transformers not found

```bash
pip install transformers torch
```

### ImportError: soniox not found

```bash
pip install soniox
```

### Soniox API errors

- Verify API key is correct
- Check internet connection
- Ensure Soniox service is available

### Memory issues with Whisper

Reduce batch size or use a smaller model variant.

## License

MIT

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tts_eval_multilingual,
  title={Multilingual TTS Evaluation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/Nightwing-77/tts_eval}
}
```
