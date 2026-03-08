"""Command-line interface for audio pair comparison."""
import argparse
import logging
import json
import sys
from pathlib import Path

from .audio_pair_comparison import AudioPair, AudioPairComparator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_files(input_path: str, output_path: str, reference_text: str) -> tuple[Path, Path, str]:
    """Validate input arguments."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")
    if not reference_text or reference_text.strip() == "":
        raise ValueError("Reference text cannot be empty")
    
    return input_file, output_file, reference_text.strip()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two audio files (input/output) using Soniox ASR and speaker embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tts_eval.cli input.wav output.wav "reference text here" --language en
  python -m tts_eval.cli input.wav output.wav "こんにちは" --language ja --api-key YOUR_KEY
        """
    )
    
    parser.add_argument("input_audio", help="Path to input audio file (WAV format)")
    parser.add_argument("output_audio", help="Path to output/generated audio file (WAV format)")
    parser.add_argument("reference_text", help="Reference transcription text")
    
    parser.add_argument("--api-key", required=False, 
                       help="Soniox API key (or set SONIOX_API_KEY env var)")
    parser.add_argument("--language", default="en",
                       help="Language code (en, ja, es, fr, de, zh, etc.) [default: en]")
    parser.add_argument("--sampling-rate", type=int, default=None,
                       help="Audio sampling rate in Hz (auto-detect if not specified)")
    parser.add_argument("--metrics", nargs="+", default=["cer", "wer"],
                       help="Metrics to compute [default: cer wer]")
    parser.add_argument("--output-json", action="store_true",
                       help="Output results as JSON instead of formatted text")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Get API key from argument or environment
    api_key = args.api_key
    if not api_key:
        import os
        api_key = os.environ.get("SONIOX_API_KEY")
    
    if not api_key:
        logger.error("Soniox API key not provided. Use --api-key or set SONIOX_API_KEY env var")
        sys.exit(1)
    
    try:
        # Validate inputs
        input_file, output_file, ref_text = validate_files(
            args.input_audio,
            args.output_audio,
            args.reference_text
        )
        
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Reference text: {ref_text}")
        logger.info(f"Language: {args.language}")
        
        # Create comparator
        logger.info("Initializing AudioPairComparator...")
        comparator = AudioPairComparator(
            soniox_api_key=api_key,
            metrics=args.metrics
        )
        
        # Create audio pair
        audio_pair = AudioPair(
            input_audio=str(input_file),
            output_audio=str(output_file),
            reference_text=ref_text,
            language=args.language,
            sampling_rate=args.sampling_rate
        )
        
        # Compare
        logger.info("Comparing audio pair...")
        result = comparator.compare(audio_pair)
        
        # Output results
        if args.output_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
