# TTS Eval
`tts_eval` is a python library for automatic evaluation of TTS outputs with Soniox-based WER/CER evaluation and speaker embedding similarity.

## Setup
```shell
pip install tts_eval
```

Set up your Soniox API key:
```shell
# Copy the example environment file
cp .env.example .env

# Edit .env to add your Soniox API key
# Get your API key from https://console.soniox.com
SONIOX_API_KEY=your_api_key_here
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

# Evaluate with strict language hints
results = evaluate_tts(
    text="hello how are you",
    language="en",  # ISO 639-1 language codes: en, ja, zh, es, fr, de, it, pt, ru, ko, ar, hi
    tts_generate_func=my_tts_function,
    metrics=["wer", "cer"]  # Calculate both WER and CER
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