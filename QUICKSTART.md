# Quick Start Guide

## 30-Second Setup

```bash
# Install with Soniox support
pip install -e ".[soniox]"
```

## Basic Usage

### Option 1: Simple ASR Evaluation (Whisper)

```python
from tts_eval import ASRMetric
import numpy as np

# Initialize
metric = ASRMetric(backend="whisper", metrics=["cer", "wer"])

# Evaluate
audio = [np.random.randn(16000)]  # Replace with real audio
transcript = "Hello world"
results = metric(audio, transcript=transcript, language="en")

print(f"CER: {results['cer'][0]:.2f}%")
print(f"WER: {results['wer'][0]:.2f}%")
```

### Option 2: Unified Pipeline (ASR + Speaker)

```python
from tts_eval import TTSEvaluationPipeline, AudioSample

# Initialize pipeline
pipeline = TTSEvaluationPipeline(
    asr_backend="soniox",  # Use Soniox for 60+ languages
    soniox_api_key="your-api-key",
    asr_metrics=["cer", "wer"]
)

# Create sample
sample = AudioSample(
    audio="tts_output.wav",
    transcript="Sample text",
    reference_speaker_audio="original_speaker.wav",
    language="en"
)

# Evaluate
result = pipeline.evaluate_sample(sample)

print(f"CER: {result.asr_scores['cer']:.2f}%")
print(f"Speaker Similarity: {result.speaker_similarity_scores['cosine_similarity']:.3f}")
```

### Option 3: Batch Evaluation with Statistics

```python
from tts_eval import TTSEvaluationPipeline, AudioSample

pipeline = TTSEvaluationPipeline(asr_backend="whisper")

# Create samples for multiple languages
samples = [
    AudioSample(audio="en_audio.wav", transcript="English text", language="en"),
    AudioSample(audio="ja_audio.wav", transcript="日本語テキスト", language="ja"),
    AudioSample(audio="es_audio.wav", transcript="Texto español", language="es"),
]

# Batch evaluate with aggregates
results = pipeline.evaluate_batch(samples, compute_aggregates=True)

# Print per-language statistics
for lang, stats in results['aggregates']['by_language'].items():
    cer_mean = stats['asr']['cer']['mean']
    print(f"{lang}: Average CER = {cer_mean:.2f}%")
```

## Choose Your Backend

### Whisper (Local, Offline)
```python
pipeline = TTSEvaluationPipeline(
    asr_backend="whisper",
    asr_model_id="kotoba-tech/kotoba-whisper-v2.0"
)
```
- Pros: No API key, offline, privacy
- Cons: Slower, requires GPU, large downloads

### Soniox (Cloud, 60+ Languages)
```python
pipeline = TTSEvaluationPipeline(
    asr_backend="soniox",
    soniox_api_key="your-api-key"
)
```
- Pros: Fast, accurate, many languages
- Cons: Requires API key, cloud-dependent

## Supported Languages

English, Japanese, Spanish, French, German, Chinese, Korean, Portuguese, Italian, Russian, Arabic, Hindi + more.

```python
from tts_eval import get_supported_languages

print(get_supported_languages())
```

## Common Tasks

### Task 1: Evaluate TTS Quality
```python
from tts_eval import TTSEvaluationPipeline, AudioSample

pipeline = TTSEvaluationPipeline(asr_backend="whisper")
sample = AudioSample(
    audio="generated.wav",
    transcript="Original text",
    language="en"
)
result = pipeline.evaluate_sample(sample)
print(f"Quality (lower CER is better): {result.asr_scores['cer']:.2f}%")
```

### Task 2: Check Speaker Preservation
```python
sample = AudioSample(
    audio="generated.wav",
    transcript="Same voice",
    reference_speaker_audio="original.wav",
    language="en"
)
result = pipeline.evaluate_sample(sample)
similarity = result.speaker_similarity_scores['cosine_similarity']
print(f"Speaker preserved: {similarity > 0.8}")
```

### Task 3: Multilingual Comparison
```python
samples = [
    AudioSample(audio=f"tts_{lang}.wav", transcript="Same text", language=lang)
    for lang in ["en", "ja", "es", "fr"]
]
results = pipeline.evaluate_batch(samples, compute_aggregates=True)

for lang in ["en", "ja", "es", "fr"]:
    cer = results['aggregates']['by_language'][lang]['asr']['cer']['mean']
    print(f"{lang}: {cer:.2f}%")
```

## Installation Options

### Minimal (Whisper only)
```bash
pip install -e .
```

### With Soniox
```bash
pip install -e ".[soniox]"
```

### Development (with testing)
```bash
pip install -e ".[dev]"
```

## Troubleshooting

**ImportError: No module named 'soniox'**
```bash
pip install soniox
```

**ImportError: No module named 'transformers'**
```bash
pip install transformers torch
```

**CUDA out of memory**
- Reduce batch size or use CPU
- Use Soniox instead (cloud-based)

**Soniox API errors**
- Check API key is valid
- Verify internet connection
- Check Soniox service status

## Next Steps

1. Read `MULTILINGUAL_FRAMEWORK.md` for detailed guide
2. Check `examples/multilingual_evaluation_example.py` for more examples
3. Run tests: `pytest tests/ -v`
4. Review `IMPLEMENTATION_SUMMARY.md` for architecture details

## API Quick Reference

```python
# Initialize
pipeline = TTSEvaluationPipeline(
    asr_backend="whisper" or "soniox",
    asr_metrics=["cer", "wer"],
    speaker_embedding_model="pyannote"
)

# Single sample
result = pipeline.evaluate_sample(sample, sample_id="123")
# Returns: EvaluationResult

# Batch samples
results = pipeline.evaluate_batch(samples, compute_aggregates=True)
# Returns: Dict with results and aggregates

# Audio sample
sample = AudioSample(
    audio="path.wav" or np.ndarray,
    transcript="text",
    reference_speaker_audio="path.wav" or np.ndarray,  # Optional
    language="en",
    sampling_rate=16000  # Optional
)

# Get language info
from tts_eval import get_language_config, get_supported_languages

config = get_language_config("en")
all_langs = get_supported_languages()
```

## Key Metrics

- **CER**: Character Error Rate (lower is better) - 0-100%
- **WER**: Word Error Rate (lower is better) - 0-100%
- **Cosine Similarity**: Speaker match (higher is better) - 0-1

**Interpretation:**
- CER/WER < 10%: Excellent
- CER/WER 10-20%: Good
- CER/WER 20-40%: Fair
- CER/WER > 40%: Poor

- Similarity > 0.8: Excellent speaker match
- Similarity 0.7-0.8: Good speaker match
- Similarity < 0.7: Poor speaker match
