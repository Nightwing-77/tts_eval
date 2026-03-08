"""Example: Compare input and output audio files using the AudioPairComparator."""
import logging
from pathlib import Path

from tts_eval import AudioPairComparator, AudioPair

# Setup logging
logging.basicConfig(level=logging.INFO)

# Your Soniox API key (get from https://soniox.com)
SONIOX_API_KEY = "your_api_key_here"

# Example 1: Simple comparison with file paths
def example_basic_comparison():
    """Basic example comparing two audio files."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Audio Pair Comparison")
    print("="*60)
    
    # Initialize the comparator
    comparator = AudioPairComparator(
        soniox_api_key=SONIOX_API_KEY,
        metrics=["cer", "wer"]
    )
    
    # Create an audio pair
    # You would replace these with actual audio file paths
    audio_pair = AudioPair(
        input_audio="path/to/input_audio.wav",      # Original/reference audio
        output_audio="path/to/output_audio.wav",    # Generated/TTS audio
        reference_text="The quick brown fox jumps over the lazy dog",
        language="en"
    )
    
    # Compare the pair
    result = comparator.compare(audio_pair)
    
    # Print results
    print(result)
    
    # Access individual results programmatically
    print(f"\nProgrammatic access:")
    print(f"Input transcription: {result.input_transcription}")
    print(f"Output transcription: {result.output_transcription}")
    print(f"CER - Input: {result.metrics['cer']['input']:.2f}%, Output: {result.metrics['cer']['output']:.2f}%")
    print(f"WER - Input: {result.metrics['wer']['input']:.2f}%, Output: {result.metrics['wer']['output']:.2f}%")
    print(f"Speaker Similarity: {result.speaker_similarity:.4f}")


# Example 2: Multilingual comparison
def example_multilingual_comparison():
    """Example comparing audio in different languages."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multilingual Comparison")
    print("="*60)
    
    comparator = AudioPairComparator(
        soniox_api_key=SONIOX_API_KEY,
        metrics=["cer", "wer"]
    )
    
    # Japanese example
    japanese_pair = AudioPair(
        input_audio="path/to/ja_input.wav",
        output_audio="path/to/ja_output.wav",
        reference_text="こんにちは、世界",
        language="ja"
    )
    
    # Spanish example
    spanish_pair = AudioPair(
        input_audio="path/to/es_input.wav",
        output_audio="path/to/es_output.wav",
        reference_text="Hola, mundo",
        language="es"
    )
    
    # Compare both
    results = [
        comparator.compare(japanese_pair),
        comparator.compare(spanish_pair)
    ]
    
    for i, result in enumerate(results, 1):
        print(f"\nLanguage {i}: {result.language}")
        print(result)


# Example 3: Batch processing
def example_batch_comparison():
    """Example comparing multiple audio pairs at once."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Audio Pair Comparison")
    print("="*60)
    
    comparator = AudioPairComparator(
        soniox_api_key=SONIOX_API_KEY,
        metrics=["cer", "wer"]
    )
    
    # Create multiple pairs
    audio_pairs = [
        AudioPair(
            input_audio=f"path/to/input_{i}.wav",
            output_audio=f"path/to/output_{i}.wav",
            reference_text=f"Reference text {i}",
            language="en"
        )
        for i in range(5)
    ]
    
    # Batch compare
    results = comparator.batch_compare(audio_pairs)
    
    # Summary statistics
    print(f"\nProcessed {len(results)} audio pairs")
    
    avg_cer_input = sum(r.metrics['cer']['input'] for r in results) / len(results)
    avg_cer_output = sum(r.metrics['cer']['output'] for r in results) / len(results)
    avg_similarity = sum(r.speaker_similarity for r in results) / len(results)
    
    print(f"Average CER (Input):  {avg_cer_input:.2f}%")
    print(f"Average CER (Output): {avg_cer_output:.2f}%")
    print(f"Average Speaker Similarity: {avg_similarity:.4f}")


# Example 4: Output as JSON
def example_json_output():
    """Example showing JSON export of results."""
    print("\n" + "="*60)
    print("EXAMPLE 4: JSON Output")
    print("="*60)
    
    import json
    
    comparator = AudioPairComparator(
        soniox_api_key=SONIOX_API_KEY,
        metrics=["cer", "wer"]
    )
    
    audio_pair = AudioPair(
        input_audio="path/to/input.wav",
        output_audio="path/to/output.wav",
        reference_text="The quick brown fox jumps over the lazy dog",
        language="en"
    )
    
    result = comparator.compare(audio_pair)
    
    # Convert to JSON
    json_result = json.dumps(result.to_dict(), indent=2)
    print("\nJSON Result:")
    print(json_result)


if __name__ == "__main__":
    # Set your API key before running examples
    if SONIOX_API_KEY == "your_api_key_here":
        print("ERROR: Please set your Soniox API key in this script first!")
        print("Get a free API key at: https://soniox.com")
        exit(1)
    
    # Uncomment the example you want to run:
    # example_basic_comparison()
    # example_multilingual_comparison()
    # example_batch_comparison()
    # example_json_output()
    
    print("This script contains examples for using AudioPairComparator.")
    print("Please see the function definitions above and uncomment to run.")
