# Quick Reference Guide

## Installation

```bash
# With Soniox support (recommended)
pip install -e ".[soniox]"

# Without Soniox (base only)
pip install -e .

# Development mode
pip install -e ".[dev]"
```

## Get API Key

Visit [soniox.com](https://soniox.com) and sign up for a free account.

## Command Line

### Basic Comparison
```bash
tts-eval input.wav output.wav "reference text here" --language en --api-key YOUR_KEY
```

### With Environment Variable
```bash
export SONIOX_API_KEY="your_key_here"
tts-eval input.wav output.wav "reference text" --language en
```

### JSON Output
```bash
tts-eval input.wav output.wav "reference text" --language en --output-json
```

### Verbose Output
```bash
tts-eval input.wav output.wav "reference text" --language en -v
```

## Python API

### Import
```python
from tts_eval import AudioPairComparator, AudioPair
```

### Single Comparison
```python
comparator = AudioPairComparator(soniox_api_key="YOUR_KEY")
pair = AudioPair(
    input_audio="input.wav",
    output_audio="output.wav",
    reference_text="text here",
    language="en"
)
result = comparator.compare(pair)
print(result)
```

### Batch Comparison
```python
pairs = [
    AudioPair("in1.wav", "out1.wav", "ref1", "en"),
    AudioPair("in2.wav", "out2.wav", "ref2", "en"),
]
results = comparator.batch_compare(pairs)
```

### Access Results
```python
# Transcriptions
print(result.input_transcription)
print(result.output_transcription)

# Metrics (WER/CER)
print(result.metrics['cer']['input'])      # Input CER%
print(result.metrics['cer']['output'])     # Output CER%
print(result.metrics['wer']['input'])      # Input WER%
print(result.metrics['wer']['output'])     # Output WER%

# Speaker similarity (0.0 to 1.0)
print(result.speaker_similarity)

# As dictionary
import json
json_str = json.dumps(result.to_dict(), indent=2)
```

## Common Patterns

### Pattern: Multilingual Evaluation
```python
comparator = AudioPairComparator(soniox_api_key="key")

for lang in ["en", "ja", "es", "fr"]:
    pair = AudioPair(
        f"input_{lang}.wav",
        f"output_{lang}.wav",
        f"reference_{lang}",
        language=lang
    )
    result = comparator.compare(pair)
    print(f"{lang}: CER={result.metrics['cer']['output']:.2f}%")
```

### Pattern: Batch with Statistics
```python
results = comparator.batch_compare(pairs)

avg_cer = sum(r.metrics['cer']['output'] for r in results) / len(results)
avg_wer = sum(r.metrics['wer']['output'] for r in results) / len(results)
avg_sim = sum(r.speaker_similarity for r in results) / len(results)

print(f"Avg CER: {avg_cer:.2f}%")
print(f"Avg WER: {avg_wer:.2f}%")
print(f"Avg Speaker Sim: {avg_sim:.4f}")
```

### Pattern: Using Numpy Arrays
```python
import numpy as np
import soundfile as sf

# Load audio
input_audio, sr = sf.read("input.wav")
output_audio, _ = sf.read("output.wav")

# Compare
pair = AudioPair(
    input_audio=input_audio,
    output_audio=output_audio,
    reference_text="text",
    language="en",
    sampling_rate=sr
)
result = comparator.compare(pair)
```

### Pattern: JSON Export
```python
import json

result = comparator.compare(pair)

# Print as JSON
print(json.dumps(result.to_dict(), indent=2))

# Save to file
with open("result.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```

## Language Codes

Common ISO 639-1 language codes:

| Code | Language | Code | Language |
|------|----------|------|----------|
| en   | English  | ja   | Japanese |
| es   | Spanish  | zh   | Chinese  |
| fr   | French   | ko   | Korean   |
| de   | German   | pt   | Portuguese |
| ru   | Russian  | ar   | Arabic   |
| hi   | Hindi    | it   | Italian  |

[Full list at soniox.com/languages](https://soniox.com/languages)

## Understanding Results

### CER/WER Interpretation
- **0-5%**: Excellent
- **5-10%**: Very Good
- **10-20%**: Good
- **20-40%**: Fair
- **>40%**: Poor

### Speaker Similarity
- **0.95-1.0**: Identical speaker
- **0.85-0.95**: Very similar speaker
- **0.70-0.85**: Similar speaker
- **0.50-0.70**: Somewhat similar
- **<0.50**: Different speaker

## Example Output

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

## Environment Variables

```bash
# Required for CLI if not using --api-key
export SONIOX_API_KEY="your_api_key_here"

# Optional: Set default language
export TTS_EVAL_LANGUAGE="en"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_audio_pair_comparison.py -v

# Run with coverage
pytest tests/ --cov=tts_eval --cov-report=html
```

## Troubleshooting

### "Soniox API key not provided"
```bash
# Option 1: Set environment variable
export SONIOX_API_KEY="your_key"

# Option 2: Pass via command line
tts-eval input.wav output.wav "text" --api-key YOUR_KEY

# Option 3: Pass in Python
AudioPairComparator(soniox_api_key="your_key")
```

### "File not found"
```bash
# Check file exists
ls -la input.wav output.wav

# Use absolute path
tts-eval /home/user/input.wav /home/user/output.wav "text" --api-key KEY
```

### "Module not found"
```bash
# Install missing dependencies
pip install soniox
pip install pyannote.audio
```

### "Language not supported"
```bash
# Check supported languages
python -c "from soniox.speech_service import SpeechClient; print('en', 'ja', 'es', etc.)"

# Use correct ISO 639-1 code
tts-eval input.wav output.wav "text" --language ja
```

## Documentation

| Document | Purpose |
|----------|---------|
| QUICKSTART.md | 30-second setup |
| AUDIO_PAIR_COMPARISON.md | Complete reference |
| IMPLEMENTATION_NOTES.md | Technical details |
| CHANGES.md | Migration guide |
| QUICK_REFERENCE.md | This file |

## Getting Help

1. **Quick questions**: Check QUICKSTART.md
2. **How to use**: Check AUDIO_PAIR_COMPARISON.md
3. **Technical details**: Check IMPLEMENTATION_NOTES.md
4. **Migrating from old version**: Check CHANGES.md
5. **Examples**: Run examples/audio_pair_comparison_example.py

## Next Steps

1. Get API key at soniox.com
2. Install: `pip install -e ".[soniox]"`
3. Try CLI: `tts-eval in.wav out.wav "text" --api-key KEY`
4. Read QUICKSTART.md (5 min)
5. Try examples: `python examples/audio_pair_comparison_example.py`

## API Endpoints

### AudioPairComparator
```python
AudioPairComparator(
    soniox_api_key: str,                    # Required
    metrics: List[str] = ["cer", "wer"],    # Optional
    speaker_embedding_model: str = "pyannote"  # Optional
)

# Methods
.compare(pair: AudioPair) -> PairComparisonResult
.batch_compare(pairs: List[AudioPair]) -> List[PairComparisonResult]
```

### AudioPair
```python
AudioPair(
    input_audio: Union[str, np.ndarray],     # Required
    output_audio: Union[str, np.ndarray],    # Required
    reference_text: str,                     # Required
    language: str = "en",                    # Optional
    sampling_rate: Optional[int] = None      # Optional
)
```

### PairComparisonResult
```python
# Properties
.input_transcription: str
.output_transcription: str
.metrics: Dict[str, Dict[str, float]]
.speaker_similarity: float
.language: str
.reference_text: str

# Methods
.to_dict() -> Dict
.__str__() -> str (pretty-printed)
```

## Tips & Tricks

### Tip 1: Use environment variable for API key
```bash
export SONIOX_API_KEY="your_key"
# Now you don't need --api-key in every command
```

### Tip 2: Save results to JSON
```bash
tts-eval in.wav out.wav "text" --output-json > result.json
```

### Tip 3: Use absolute paths
```bash
tts-eval /absolute/path/input.wav /absolute/path/output.wav "text" --api-key KEY
```

### Tip 4: Debug with verbose mode
```bash
tts-eval in.wav out.wav "text" --language en -v
```

### Tip 5: Process multiple pairs programmatically
```python
for i in range(10):
    pair = AudioPair(f"in_{i}.wav", f"out_{i}.wav", f"ref_{i}", "en")
    result = comparator.compare(pair)
    # Process result
```

## Performance Tips

1. Use Soniox (cloud-based, no GPU needed)
2. Process in batches when possible
3. Cache results to avoid reprocessing
4. Use multi-threading for batch processing
5. Export results to JSON for archival

---

**For more information, see the complete documentation in AUDIO_PAIR_COMPARISON.md**
