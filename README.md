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
from transformers import AutoModelForCausalLM

# Load your TTS model
tts_model = AutoModelForCausalLM.from_pretrained("your-tts-model")

# Evaluate with strict language hints
results = evaluate_tts(
    text="hello how are you",
    language="en",  # Strict language hint: en, ja, zh, es, fr, de, it, pt, ru, ko, ar, hi
    tts_model=tts_model,
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
from transformers import AutoModelForCausalLM

# Load your TTS model
tts_model = AutoModelForCausalLM.from_pretrained("your-tts-model")

# Create evaluator instance
evaluator = UnifiedTTSEvaluator(
    tts_model=tts_model,
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
Strict language hints are supported for:
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
- Soniox console CLI installed
- Python 3.8+
- transformers (for TTS model loading)
- torch, numpy, librosa, soundfile