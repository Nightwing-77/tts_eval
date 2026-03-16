# TTS Evaluation Library

A unified TTS evaluation library that combines Soniox WER/CER evaluation with speaker embedding similarity for comprehensive TTS model assessment.

## Installation

### Option 1: Git Clone (Recommended)
```bash
# Clone the repository
git clone https://github.com/Nightwing-77/tts_eval
cd tts_eval

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Soniox API key
```

### Option 2: Pip Install
```bash
pip install tts-eval
# Still need to set SONIOX_API_KEY environment variable
```

## Setup

Set up your Soniox API key:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to add your Soniox API key
# Get your API key from https://console.soniox.com
SONIOX_API_KEY=your_api_key_here
```

## Quick Start

### Basic Usage

```python
from tts_eval import evaluate_tts

def my_tts_function(text: str, language: str):
    """
    Your TTS function that takes text and language and returns audio array.
    This would contain your Svara-TTS model setup and audio generation logic.
    """
    # Your Svara-TTS code here:
    # voice = f"{language} (Female)"  # or Male
    # formatted_text = f"<|audio|> {voice}: {text}<|eot_id|>"
    # ... rest of your Svara-TTS generation logic
    # return audio_array
    pass

# Basic evaluation
results = evaluate_tts(
    text="hello how are you",
    language="en",
    tts_generate_func=my_tts_function,
    metrics=["wer", "cer"]
)

print(results)
# {
#   'wer': 0.0,
#   'cer': 0.0,
#   'transcription': 'hello how are you',
#   'input_text': 'hello how are you',
#   'language': 'en',
#   'soniox_language_hint': 'en'
# }
```

### Advanced Usage with Speaker Similarity

```python
# Evaluation with speaker similarity
results = evaluate_tts(
    text="hello how are you",
    language="en",
    tts_generate_func=my_tts_function,
    reference_audio="reference_speech.wav",  # Add reference audio
    metrics=["wer", "cer", "similarity"]
)

print(results)
# {
#   'wer': 0.0,
#   'cer': 0.0,
#   'speaker_similarity': 0.85,
#   'transcription': 'hello how are you',
#   'input_text': 'hello how are you',
#   'language': 'en',
#   'soniox_language_hint': 'en'
# }
```

### Using the UnifiedTTSEvaluator Class

```python
from tts_eval import UnifiedTTSEvaluator

# Initialize the evaluator
evaluator = UnifiedTTSEvaluator(
    tts_generate_func=my_tts_function,
    speaker_embedding_model="metavoice"  # or "pyannote", "clap", etc.
)

# Evaluate multiple texts
texts = ["hello world", "how are you today", "this is a test"]
for text in texts:
    results = evaluator.evaluate(
        text=text,
        language="en",
        reference_audio="reference.wav",  # Optional
        metrics=["wer", "cer", "similarity"]
    )
    print(f"{text}: WER={results['wer']:.1f}, Similarity={results.get('speaker_similarity', 'N/A')}")
```

## Unified TTS Evaluation

### Main Evaluation Function
The unified evaluator provides a single interface for comprehensive TTS evaluation with strict Soniox language hints.

***Simple Usage:***

```python
from tts_eval import evaluate_tts

# Define your TTS generation function (e.g., for Svara-TTS)
def my_tts_function(text: str, language: str):
    """
    Your TTS function that takes text and language and returns audio array.
    This would contain your Svara-TTS model setup and audio generation logic.
    """
    # Your Svara-TTS code here:
    # voice = f"{language} (Female)"  # or Male
    # formatted_text = f"<|audio|> {voice}: {text}<|eot_id|>"
    # ... rest of your Svara-TTS generation logic
    # return audio_array
    pass

# Evaluate with strict language hints and speaker similarity
results = evaluate_tts(
    text="hello how are you",
    language="en",  # ISO 639-1 language codes: en, ja, zh, es, fr, de, it, pt, ru, ko, ar, hi
    tts_generate_func=my_tts_function,
    reference_audio="reference_speech.wav",  # Add reference audio for speaker similarity
    metrics=["wer", "cer", "similarity"]  # Include similarity metric
)

print(results)
# {
#   'wer': 5.2,
#   'cer': 3.1,
#   'transcription': 'hello how are you',
#   'input_text': 'hello how are you',
#   'language': 'en',
#   'soniox_language_hint': 'english'
# }
```

***Advanced Usage with Speaker Similarity:***

```python
from tts_eval import UnifiedTTSEvaluator

# Define your TTS generation function
def svara_tts_function(text: str, language: str):
    """
    Svara-TTS generation function that returns audio array at 24kHz.
    """
    # Your Svara-TTS implementation here
    # voice = f"{language} (Male)"
    # formatted_text = f"<|audio|> {voice}: {text}<|eot_id|>"
    # ... SNAC decoding logic
    # return audio_array
    pass

# Create evaluator instance
evaluator = UnifiedTTSEvaluator(
    tts_generate_func=svara_tts_function,
    speaker_embedding_model="metavoice"
)

# Evaluate with speaker similarity
results = evaluator.evaluate(
    text="hello how are you",
    language="en",
    reference_audio="reference_speech.wav",  # For speaker similarity
    metrics=["wer", "cer", "similarity"]
)

print(results)
# {
#   'wer': 5.2,
#   'cer': 3.1,
#   'speaker_similarity': 0.85,
#   'transcription': 'hello how are you',
#   'generated_audio_path': '/tmp/tmp_xyz.wav',
#   'input_text': 'hello how are you',
#   'language': 'en',
#   'soniox_language_hint': 'english'
# }
```

### Individual Components

#### Soniox WER/CER Metric
Direct access to Soniox-based WER/CER evaluation with strict language hints.

```python
from tts_eval import evaluate_wer_with_language_hint

# Evaluate existing audio files
results = evaluate_wer_with_language_hint(
    audio_files=["audio1.wav", "audio2.flac"],
    reference_transcript="Hello world, this is a test",
    language="en",
    metrics=["wer", "cer"]
)

print(results)
# {'wer': [5.2, 8.1], 'cer': [3.1, 4.7]}
```

#### Speaker Embedding Similarity
Speaker embedding similarity for voice cloning evaluation.

```python
from tts_eval import SpeakerEmbeddingSimilarity

pipe = SpeakerEmbeddingSimilarity(model_id="metavoice")
output = pipe(
    audio_target=["sample_1.flac", "sample_2.wav"],
    audio_reference="sample_1.flac"
)
print(output)
# {'cosine_similarity': [1.0000001, 0.65718323]}
```

***Available Speaker Embedding Models:***
- `metavoice` - MetaVoice speaker embeddings
- `pyannote` - Pyannote speaker embeddings  
- `clap`, `clap_general` - CLAP audio embeddings
- `w2v_bert` - Wav2Vec-BERT embeddings
- `hubert_xl`, `hubert_large`, `hubert_base` - HuBERT embeddings
- `wav2vec` - Wav2Vec embeddings
- `xlsr_2b`, `xlsr_1b`, `xlsr_300m` - XLSR embeddings

## Supported Languages for Soniox
Strict language hints are supported using ISO 639-1 codes:
- English (`en`)
- Japanese (`ja`) 
- Chinese (`zh`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Italian (`it`)
- Portuguese (`pt`)
- Russian (`ru`)
- Korean (`ko`)
- Arabic (`ar`)
- Hindi (`hi`)

## Requirements
- Soniox API key (get from https://console.soniox.com)
- Soniox Python SDK (`pip install soniox`)
- Python 3.8+
- torch, numpy, librosa, soundfile, python-dotenv
- Your TTS model/function (e.g., Svara-TTS with SNAC)

## TTS Function Interface
The evaluator expects a function with this signature:
```python
def your_tts_function(text: str, language: str) -> numpy.ndarray:
    """
    Generate audio from text and return as numpy array.
    
    Args:
        text: Input text to synthesize
        language: Language code (e.g., 'en', 'ja', 'zh')
        
    Returns:
        Audio array (numpy.ndarray) at your model's sample rate
    """
    # Your TTS generation logic here
    return audio_array
```

For Svara-TTS users, wrap your existing `generate_audio_from_text()` function to match this interface.