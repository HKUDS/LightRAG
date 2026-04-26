"""
Download all necessary cache files for offline deployment.

This module provides a CLI command to download tiktoken model cache files
for offline environments where internet access is not available.
"""

import os
import sys
from pathlib import Path


# Known tiktoken encoding names (not model names)
# These need to be loaded with tiktoken.get_encoding() instead of tiktoken.encoding_for_model()
TIKTOKEN_ENCODING_NAMES = {"cl100k_base", "p50k_base", "r50k_base", "o200k_base"}


def download_tiktoken_cache(cache_dir: str = None, models: list = None):
    """Download tiktoken models to local cache

    Args:
        cache_dir: Directory to store the cache files. If None, uses tiktoken's default location.
        models: List of model names or encoding names to download. If None, downloads common ones.

    Returns:
        Tuple of (success_count, failed_models, actual_cache_dir)
    """
    # If user specified a cache directory, set it BEFORE importing tiktoken
    # tiktoken reads TIKTOKEN_CACHE_DIR at import time
    user_specified_cache = cache_dir is not None

    if user_specified_cache:
        cache_dir = os.path.abspath(cache_dir)
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"Using specified cache directory: {cache_dir}")
    else:
        # Check if TIKTOKEN_CACHE_DIR is already set in environment
        env_cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
        if env_cache_dir:
            cache_dir = env_cache_dir
            print(f"Using TIKTOKEN_CACHE_DIR from environment: {cache_dir}")
        else:
            # Use tiktoken's default location (tempdir/data-gym-cache)
            import tempfile

            cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
            print(f"Using tiktoken default cache directory: {cache_dir}")

    # Now import tiktoken (it will use the cache directory we determined)
    try:
        import tiktoken
    except ImportError:
        print("Error: tiktoken is not installed.")
        print("Install with: pip install tiktoken")
        sys.exit(1)

    # Common models used by LightRAG and OpenAI
    if models is None:
        models = [
            "gpt-4o-mini",  # Default model for LightRAG
            "gpt-4o",  # GPT-4 Omni
            "gpt-4",  # GPT-4
            "gpt-3.5-turbo",  # GPT-3.5 Turbo
            "text-embedding-ada-002",  # Legacy embedding model
            "text-embedding-3-small",  # Small embedding model
            "text-embedding-3-large",  # Large embedding model
            "cl100k_base",  # Default encoding for LightRAG
        ]

    print(f"\nDownloading {len(models)} tiktoken models...")
    print("=" * 70)

    success_count = 0
    failed_models = []

    for i, model in enumerate(models, 1):
        try:
            print(f"[{i}/{len(models)}] Downloading {model}...", end=" ", flush=True)
            # Use get_encoding for encoding names, encoding_for_model for model names
            if model in TIKTOKEN_ENCODING_NAMES:
                encoding = tiktoken.get_encoding(model)
            else:
                encoding = tiktoken.encoding_for_model(model)
            # Trigger download by encoding a test string
            encoding.encode("test")
            print("✓ Done")
            success_count += 1
        except KeyError as e:
            print(f"✗ Failed: Unknown model or encoding '{model}'")
            failed_models.append((model, str(e)))
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed_models.append((model, str(e)))

    print("=" * 70)
    print(f"\n✓ Successfully cached {success_count}/{len(models)} models")

    if failed_models:
        print(f"\n✗ Failed to download {len(failed_models)} models:")
        for model, error in failed_models:
            print(f"  - {model}: {error}")

    print(f"\nCache location: {cache_dir}")
    print("\nFor offline deployment:")
    print("  1. Copy directory to offline server:")
    print(f"     tar -czf tiktoken_cache.tar.gz {cache_dir}")
    print("     scp tiktoken_cache.tar.gz user@offline-server:/path/to/")
    print("")
    print("  2. On offline server, extract and set environment variable:")
    print("     tar -xzf tiktoken_cache.tar.gz")
    print("     export TIKTOKEN_CACHE_DIR=/path/to/tiktoken_cache")
    print("")
    print("  3. Or copy to default location:")
    print(f"     cp -r {cache_dir} ~/.tiktoken_cache/")

    return success_count, failed_models


def main():
    """Main entry point for the CLI command"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="lightrag-download-cache",
        description="Download cache files for LightRAG offline deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location (~/.tiktoken_cache)
  lightrag-download-cache

  # Download to specific directory
  lightrag-download-cache --cache-dir ./offline_cache/tiktoken

  # Download specific models only
  lightrag-download-cache --models gpt-4o-mini gpt-4

For more information, visit: https://github.com/HKUDS/LightRAG
        """,
    )

    parser.add_argument(
        "--cache-dir",
        help="Cache directory path (default: ~/.tiktoken_cache)",
        default=None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to download (default: common models)",
        default=None,
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s (LightRAG cache downloader)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LightRAG Offline Cache Downloader")
    print("=" * 70)

    try:
        success_count, failed_models = download_tiktoken_cache(
            args.cache_dir, args.models
        )

        print("\n" + "=" * 70)
        print("Download Complete")
        print("=" * 70)

        # Exit with error code if all downloads failed
        if success_count == 0:
            print("\n✗ All downloads failed. Please check your internet connection.")
            sys.exit(1)
        # Exit with warning code if some downloads failed
        elif failed_models:
            print(
                f"\n⚠ Some downloads failed ({len(failed_models)}/{success_count + len(failed_models)})"
            )
            sys.exit(2)
        else:
            print("\n✓ All cache files downloaded successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n✗ Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
