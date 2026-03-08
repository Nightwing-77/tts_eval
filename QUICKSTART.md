# TTS Evaluation Framework - Quick Start Guide

Evaluate and compare TTS output audio against input audio using Soniox ASR and speaker embeddings.

## Installation

```bash
# Install the package with Soniox support
pip install -e ".[soniox]"

# Or just install core dependencies
pip install -e .
```

## Get Your Soniox API Key

1. Visit [soniox.com](https://soniox.com)
2. Sign up for a free account
3. Get your API key from the dashboard

## Quick Start (30 seconds)

### Option 1: Using the CLI

Compare two audio files with a single command:

```bash
tts-eval input_audio.wav output_audio.wav "reference text here" --language en --api-key YOUR_API_KEY
```

### Option 2: Using Python

```python
from tts_eval import AudioPairComparator, AudioPair

# Initialize comparator
comparator = AudioPairComparator(soniox_api_key="your_api_key")

# Create audio pair
pair = AudioPair(
    input_audio="input.wav",
    output_audio="output.wav", 
    reference_text="reference text here",
    language="en"
)

# Compare
result = comparator.compare(pair)

# View results
print(result)
```

## Results Explained

The output shows:

- **Transcriptions**: What Soniox transcribed from each audio
- **WER/CER**: Word Error Rate / Character Error Rate (lower is better)
  - Compared against reference text
  - Shows improvement between input and output
- **Speaker Similarity**: Cosine similarity of speaker embeddings (1.0 = identical, 0.0 = different)

### Example Output

```
============================================================
AUDIO PAIR COMPARISON RESULTS
============================================================

Language: en
Reference Text: the quick brown fox jumps

--- TRANSCRIPTIONS ---
Input Transcription:  the quick brown fox jumps
Output Transcription: the quick brown fox jumps

--- METRICS (WER/CER) ---
CER:
  Input:  0.00%
  Output: 0.00%
  Change: = 0.00%

WER:
  Input:  0.00%
  Output: 0.00%
  Change: = 0.00%

--- SPEAKER SIMILARITY ---
Cosine Similarity: 0.9234
(1.0 = identical speaker, 0.0 = completely different)

============================================================
```

## Common Tasks

### Compare Multiple Language Samples

```python
comparator = AudioPairComparator(soniox_api_key="key")

languages = {
    "en": ("input_en.wav", "output_en.wav", "english text"),
    "ja": ("input_ja.wav", "output_ja.wav", "日本語テキスト"),
    "es": ("input_es.wav", "output_es.wav", "texto español"),
}

for lang, (inp, out, ref) in languages.items():
    pair = AudioPair(inp, out, ref, language=lang)
    result = comparator.compare(pair)
    print(f"\n{lang.upper()}: WER={result.metrics['wer']['output']:.2f}%")
```

### Batch Process Multiple Pairs

```python
pairs = [
    AudioPair("in1.wav", "out1.wav", "ref1", "en"),
    AudioPair("in2.wav", "out2.wav", "ref2", "en"),
    AudioPair("in3.wav", "out3.wav", "ref3", "en"),
]

results = comparator.batch_compare(pairs)

# Summary
avg_cer = sum(r.metrics['cer']['output'] for r in results) / len(results)
avg_similarity = sum(r.speaker_similarity for r in results) / len(results)

print(f"Average CER: {avg_cer:.2f}%")
print(f"Average Speaker Similarity: {avg_similarity:.4f}")
```

### Export Results as JSON

```python
import json

result = comparator.compare(pair)
json_data = json.dumps(result.to_dict(), indent=2)
print(json_data)

# Save to file
with open("results.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```

### Using Numpy Arrays Instead of Files

```python
import numpy as np

# Generate or load audio arrays
input_audio = np.random.randn(16000)  # 1 second at 16kHz
output_audio = np.random.randn(16000)

pair = AudioPair(
    input_audio=input_audio,
    output_audio=output_audio,
    reference_text="your text",
    language="en",
    sampling_rate=16000  # Important!
)

result = comparator.compare(pair)
```

## Supported Languages

Soniox supports 60+ languages. Common ones:

- `en` - English
- `ja` - Japanese
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German
- `ko` - Korean
- `pt` - Portuguese
- `ru` - Russian
- `ar` - Arabic

[See full list at soniox.com/languages](https://soniox.com)

## API Reference

### AudioPairComparator

```python
comparator = AudioPairComparator(
    soniox_api_key: str,           # Required: Soniox API key
    metrics: List[str] = ["cer", "wer"],  # Metrics to compute
    speaker_embedding_model: str = "pyannote"  # Speaker model
)
```

### AudioPair

```python
pair = AudioPair(
    input_audio: Union[str, np.ndarray],   # Path or audio array
    output_audio: Union[str, np.ndarray],  # Path or audio array
    reference_text: str,                   # Text to compare against
    language: str = "en",                  # Language code
    sampling_rate: Optional[int] = None    # Sampling rate (needed for arrays)
)
```

### PairComparisonResult

```python
result.input_transcription   # str - Soniox transcription of input
result.output_transcription  # str - Soniox transcription of output
result.metrics              # Dict - {"cer": {"input": 5.0, "output": 3.0}, ...}
result.speaker_similarity   # float - 0.0 to 1.0
result.language            # str - Language code
result.reference_text      # str - Reference text used

result.to_dict()           # Convert to dictionary
print(result)             # Pretty-printed results
```

## Environment Variables

Instead of passing `--api-key` every time, set an environment variable:

```bash
export SONIOX_API_KEY="your_api_key_here"

# Now you can omit --api-key
tts-eval input.wav output.wav "text" --language en
```

## Troubleshooting

### "Soniox API key not provided"
- Set `SONIOX_API_KEY` environment variable, or
- Pass `--api-key YOUR_KEY` to CLI, or
- Pass `soniox_api_key="YOUR_KEY"` to Python

### "File not found"
- Check that audio file paths are correct
- Use absolute paths or relative from current working directory

### "Audio format not supported"
- Ensure audio files are WAV format
- For numpy arrays, ensure `sampling_rate` is specified

### "Language not supported"
- Check the [Soniox language list](https://soniox.com/languages)
- Use correct ISO 639-1 language code

## Examples

See the `examples/` directory for more:

```bash
python examples/audio_pair_comparison_example.py
```

## More Information

- [Full Documentation](MULTILINGUAL_FRAMEWORK.md)
- [GitHub Repository](https://github.com/Nightwing-77/tts_eval)
- [Soniox Docs](https://soniox.com/docs)
