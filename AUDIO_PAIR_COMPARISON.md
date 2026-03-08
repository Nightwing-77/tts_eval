# Audio Pair Comparison Framework

A Python framework for comparing input/output audio pairs using Soniox ASR (60+ languages) and speaker embeddings.

## Overview

The Audio Pair Comparison Framework enables:

1. **Soniox-based ASR Transcription** - Convert audio to text in 60+ languages
2. **WER/CER Metrics** - Compare transcription accuracy against reference text
3. **Speaker Similarity Analysis** - Measure if speaker identity is preserved between input and output
4. **Batch Processing** - Evaluate multiple audio pairs simultaneously
5. **Multiple Output Formats** - View results as formatted text or JSON

## Architecture

### Core Components

#### 1. ASRMetric (tts_eval/metric_asr.py)

Simplified Soniox-only transcription engine:

```python
from tts_eval import ASRMetric

asr = ASRMetric(api_key="your_soniox_key", metrics=["cer", "wer"])

# Transcribe single audio
text = asr.transcribe("audio.wav", language="en")

# Compute metrics
metrics = asr.compute_metrics(predicted_text, reference_text)
# Returns: {"cer": 5.2, "wer": 10.1}
```

**Features:**
- Single API: `transcribe()` and `compute_metrics()`
- Supports file paths and numpy arrays
- Language code mapping for Soniox
- Automatic WAV conversion for arrays

#### 2. SpeakerEmbeddingSimilarity (tts_eval/metric_speaker_embedding_similarity.py)

Speaker embedding comparison using Pyannote:

```python
from tts_eval import SpeakerEmbeddingSimilarity

speaker_sim = SpeakerEmbeddingSimilarity(model_id="pyannote")

# Compare speaker embeddings
similarity = speaker_sim(
    [audio1],
    [audio2],
    sampling_rate=16000
)
# Returns: [0.95]  # 1.0 = identical speaker
```

#### 3. AudioPairComparator (tts_eval/audio_pair_comparison.py)

Main pipeline combining ASR + speaker similarity:

```python
from tts_eval import AudioPairComparator, AudioPair

comparator = AudioPairComparator(soniox_api_key="key")

# Create audio pair
pair = AudioPair(
    input_audio="input.wav",        # Original/reference audio
    output_audio="output.wav",      # Generated/TTS audio
    reference_text="text here",     # Ground truth transcription
    language="en"
)

# Compare
result = comparator.compare(pair)

# Access results
print(f"Input transcription: {result.input_transcription}")
print(f"Output transcription: {result.output_transcription}")
print(f"CER improvement: {result.metrics['cer']['input'] - result.metrics['cer']['output']:.2f}%")
print(f"Speaker similarity: {result.speaker_similarity:.4f}")
```

**Methods:**
- `compare(pair)` - Compare single audio pair
- `batch_compare(pairs)` - Compare multiple pairs

## Data Structures

### AudioPair

Represents input/output audio to compare:

```python
@dataclass
class AudioPair:
    input_audio: Union[str, np.ndarray]    # File path or audio array
    output_audio: Union[str, np.ndarray]   # File path or audio array
    reference_text: str                     # Ground truth transcription
    language: str = "en"                    # Language code
    sampling_rate: Optional[int] = None    # For numpy arrays (default: 16000)
```

### PairComparisonResult

Result of comparing two audio files:

```python
@dataclass
class PairComparisonResult:
    input_transcription: str               # Soniox transcription of input
    output_transcription: str              # Soniox transcription of output
    metrics: Dict[str, Dict[str, float]]   # {"cer": {"input": 5.0, "output": 3.0}, ...}
    speaker_similarity: float              # 0.0 to 1.0
    language: str                          # Language code
    reference_text: str                    # Reference text used
```

**Methods:**
- `to_dict()` - Convert to dictionary for JSON serialization
- `__str__()` - Pretty-printed formatted output

## Usage Examples

### Example 1: Basic Comparison

```python
from tts_eval import AudioPairComparator, AudioPair

comparator = AudioPairComparator(soniox_api_key="YOUR_KEY")

pair = AudioPair(
    input_audio="speaker_original.wav",
    output_audio="speaker_generated.wav",
    reference_text="The quick brown fox",
    language="en"
)

result = comparator.compare(pair)
print(result)  # Pretty-formatted results
```

### Example 2: Batch Processing with Statistics

```python
import numpy as np

# Create 10 audio pairs
pairs = [
    AudioPair(
        input_audio=f"input_{i}.wav",
        output_audio=f"output_{i}.wav",
        reference_text="Reference text",
        language="en"
    )
    for i in range(10)
]

# Batch compare
results = comparator.batch_compare(pairs)

# Calculate averages
avg_cer = sum(r.metrics['cer']['output'] for r in results) / len(results)
avg_sim = sum(r.speaker_similarity for r in results) / len(results)

print(f"Batch Results:")
print(f"  Average CER: {avg_cer:.2f}%")
print(f"  Average Speaker Similarity: {avg_sim:.4f}")
```

### Example 3: Multilingual Evaluation

```python
languages_config = {
    "en": ("en_in.wav", "en_out.wav", "English reference"),
    "ja": ("ja_in.wav", "ja_out.wav", "日本語リファレンス"),
    "es": ("es_in.wav", "es_out.wav", "Español referencia"),
    "fr": ("fr_in.wav", "fr_out.wav", "Référence français"),
}

comparator = AudioPairComparator(soniox_api_key="KEY")
results_by_lang = {}

for lang, (inp, out, ref) in languages_config.items():
    pair = AudioPair(inp, out, ref, language=lang)
    result = comparator.compare(pair)
    results_by_lang[lang] = result

# Print summary by language
for lang, result in results_by_lang.items():
    cer_output = result.metrics['cer']['output']
    sim = result.speaker_similarity
    print(f"{lang:2s}: CER={cer_output:6.2f}%  Speaker Sim={sim:.4f}")
```

### Example 4: JSON Export

```python
import json

result = comparator.compare(pair)

# Convert to JSON
json_result = json.dumps(result.to_dict(), indent=2)
print(json_result)

# Save to file
with open("evaluation_result.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```

### Example 5: Using Numpy Arrays

```python
import numpy as np
import soundfile as sf

# Load audio as numpy arrays
input_audio, sr_in = sf.read("input.wav")
output_audio, sr_out = sf.read("output.wav")

# Both should have same sampling rate
assert sr_in == sr_out

pair = AudioPair(
    input_audio=input_audio,
    output_audio=output_audio,
    reference_text="Text here",
    language="en",
    sampling_rate=sr_in
)

result = comparator.compare(pair)
```

## CLI Usage

### Basic Command

```bash
tts-eval input.wav output.wav "reference text" --language en --api-key YOUR_KEY
```

### With Environment Variable

```bash
export SONIOX_API_KEY="YOUR_KEY"
tts-eval input.wav output.wav "reference text" --language en
```

### JSON Output

```bash
tts-eval input.wav output.wav "reference text" --language en --output-json
```

### Verbose Logging

```bash
tts-eval input.wav output.wav "reference text" --language en -v
```

## Supported Languages

Soniox supports 60+ languages with ISO 639-1 codes:

| Code | Language | Code | Language |
|------|----------|------|----------|
| en   | English  | ja   | Japanese |
| es   | Spanish  | zh   | Chinese  |
| fr   | French   | ko   | Korean   |
| de   | German   | pt   | Portuguese |
| ru   | Russian  | ar   | Arabic   |
| hi   | Hindi    | it   | Italian  |
| pl   | Polish   | nl   | Dutch    |
| tr   | Turkish  | th   | Thai     |

[Full list at soniox.com/languages](https://soniox.com/languages)

## Metrics Explained

### Character Error Rate (CER)
- **Formula**: (Substitutions + Deletions + Insertions) / Reference Length × 100
- **Range**: 0-100% (lower is better)
- **Example**: 5% CER means 5 character errors per 100 characters

### Word Error Rate (WER)
- **Formula**: (Substitutions + Deletions + Insertions) / Reference Words × 100
- **Range**: 0-100% (lower is better)
- **Example**: 10% WER means 10 word errors per 100 words

### Speaker Similarity (Cosine Similarity)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 1.0 = Identical speaker
  - 0.8+ = Very similar speaker
  - 0.5-0.8 = Similar speaker
  - <0.5 = Different speaker

## Performance Benchmarks

On typical hardware:
- **Soniox transcription**: 10-30 seconds per minute of audio
- **Speaker embedding**: 2-5 seconds per audio file
- **Metrics computation**: <100ms per pair
- **Batch overhead**: ~2 seconds per 100 pairs

## Error Handling

```python
from tts_eval import AudioPairComparator, AudioPair

try:
    comparator = AudioPairComparator(soniox_api_key="KEY")
except ImportError as e:
    print(f"Missing dependency: {e}")

try:
    pair = AudioPair("nonexistent.wav", "output.wav", "text", "en")
    result = comparator.compare(pair)
except FileNotFoundError as e:
    print(f"File error: {e}")
except Exception as e:
    print(f"Transcription error: {e}")
```

## Installation

### Full Installation

```bash
pip install -e ".[soniox]"
```

### Core Only (speaker embedding + utils)

```bash
pip install -e .
```

### Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Dependencies

### Required
- numpy
- evaluate (for WER/CER computation)
- soundfile (for audio I/O)
- jiwer (for metric calculation)
- pyannote.audio (for speaker embeddings)

### Optional
- soniox (for ASR transcription)

## Troubleshooting

### Import Errors

```python
# If you get: ImportError: No module named 'soniox'
pip install soniox

# If you get: ImportError: No module named 'pyannote'
pip install pyannote.audio
```

### File Not Found

```bash
# Use absolute paths or check current working directory
pwd
ls input.wav  # Verify file exists

# Or specify full path
tts-eval /home/user/input.wav /home/user/output.wav "text" --language en
```

### API Key Issues

```bash
# Verify API key
echo $SONIOX_API_KEY

# Test API connection
python -c "from soniox.speech_service import SpeechClient; c = SpeechClient(api_key='YOUR_KEY'); print('OK')"
```

## API Reference

### AudioPairComparator

```python
class AudioPairComparator:
    def __init__(
        self,
        soniox_api_key: str,
        metrics: Union[str, List[str]] = ["cer", "wer"],
        speaker_embedding_model: str = "pyannote"
    ):
        """Initialize comparator"""

    def compare(self, audio_pair: AudioPair) -> PairComparisonResult:
        """Compare single audio pair"""

    def batch_compare(
        self,
        audio_pairs: List[AudioPair]
    ) -> List[PairComparisonResult]:
        """Compare multiple audio pairs"""
```

### ASRMetric

```python
class ASRMetric:
    def __init__(
        self,
        api_key: str,
        metrics: Union[str, List[str]] = "cer"
    ):
        """Initialize Soniox ASR"""

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: str = "en",
        sampling_rate: Optional[int] = None
    ) -> str:
        """Transcribe audio file"""

    def compute_metrics(
        self,
        predicted_text: str,
        reference_text: str,
        normalize: bool = True
    ) -> Dict[str, float]:
        """Compute WER/CER metrics"""
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Make changes and test (`pytest tests/`)
4. Commit and push
5. Open pull request

## License

MIT License - see LICENSE file

## Support

- Issues: [GitHub Issues](https://github.com/Nightwing-77/tts_eval/issues)
- Documentation: [docs/](docs/)
- Examples: [examples/](examples/)
- API Key: [soniox.com](https://soniox.com)
