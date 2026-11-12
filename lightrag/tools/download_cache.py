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

# spaCy language models used by the native docx `smart_heading` engine parameter.
# Pinned to an exact version: smart_heading promises deterministic re-parse
# results across environments, and a model drift would silently change NER /
# sentence-split decisions. Keep in sync with requirements-offline-smart-heading.txt.
# spacy-pkuseg is zh_core_web_sm's tokenizer backend (a PyPI dependency the
# model wheel does not bundle); it is pinned and shipped with the wheels for
# the same determinism promise — without it the offline install of the zh
# model from this wheel directory cannot resolve its dependency.
SPACY_MODEL_WHEELS = {
    "zh_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl",
    "en_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    "spacy-pkuseg": "spacy-pkuseg==1.0.2",
}


def _spacy_installer_command(install: bool) -> list[str] | None:
    """Return the command prefix that installs or downloads one spaCy wheel.

    pip of the running interpreter is preferred. A virtual environment created
    by ``uv venv`` / ``uv sync`` ships without pip, so for ``install`` the
    ``uv`` on PATH is used instead, targeted at this interpreter with
    ``--python`` so the models land in the same environment. ``uv`` has no
    ``pip download`` equivalent, so downloading still requires pip.

    Returns None when neither tool is usable; the caller must report that.
    """
    import importlib.util
    import shutil

    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install" if install else "download"]
    if install:
        uv = shutil.which("uv")
        if uv:
            return [uv, "pip", "install", "--python", sys.executable]
    return None


def download_spacy_models(spacy_dir: str = None, install: bool = False):
    """Download (and optionally install) the pinned spaCy model wheels.

    Args:
        spacy_dir: Directory to store the downloaded wheels. Defaults to
            ``./spacy_models``. Ignored when ``install`` is True.
        install: If True, install the wheels into the current environment
            instead of downloading them to a directory. Uses pip, or
            ``uv pip install`` when the environment has no pip (see
            ``_spacy_installer_command``).

    Returns:
        Tuple of (success_count, failed_models)
    """
    import subprocess

    success_count = 0
    failed_models = []

    base_cmd = _spacy_installer_command(install)
    if base_cmd is None:
        if install:
            reason = (
                "neither pip nor uv is available for this Python environment; "
                "install one of them, or run: "
                "uv pip install -r requirements-offline-smart-heading.txt"
            )
        else:
            reason = (
                "downloading wheels requires pip, which this environment lacks "
                "(uv-created virtual environments ship without it); add it with "
                "`uv pip install pip` or `python -m ensurepip`"
            )
        print(f"\n✗ Cannot process spaCy models: {reason}")
        return 0, [(name, reason) for name in sorted(SPACY_MODEL_WHEELS)]

    if install:
        print(f"\nInstalling {len(SPACY_MODEL_WHEELS)} spaCy models...")
    else:
        spacy_dir = os.path.abspath(spacy_dir or "./spacy_models")
        Path(spacy_dir).mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {len(SPACY_MODEL_WHEELS)} spaCy model wheels...")
        print(f"Using spaCy model directory: {spacy_dir}")
    print("=" * 70)

    for i, (name, url) in enumerate(sorted(SPACY_MODEL_WHEELS.items()), 1):
        if install:
            cmd = [*base_cmd, url]
            action = "Installing"
        else:
            cmd = [*base_cmd, "--no-deps", "--dest", spacy_dir, url]
            action = "Downloading"
        try:
            print(
                f"[{i}/{len(SPACY_MODEL_WHEELS)}] {action} {name}...",
                end=" ",
                flush=True,
            )
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Done")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print("✗ Failed")
            detail = (e.stderr or e.stdout or str(e)).strip().splitlines()
            failed_models.append((name, detail[-1] if detail else str(e)))

    print("=" * 70)
    print(
        f"\n✓ Successfully processed {success_count}/{len(SPACY_MODEL_WHEELS)} spaCy models"
    )

    if failed_models:
        print(f"\n✗ Failed spaCy models ({len(failed_models)}):")
        for name, error in failed_models:
            print(f"  - {name}: {error}")

    if not install:
        print(f"\nspaCy model wheels location: {spacy_dir}")
        print("\nFor offline deployment:")
        print("  1. Copy the wheels to the offline server together with")
        print("     requirements-offline-smart-heading.txt")
        print("  2. On the offline server install them with:")
        print(f"     pip install --no-index --find-links={spacy_dir} \\")
        print("         zh_core_web_sm en_core_web_sm")

    return success_count, failed_models


def download_tiktoken_cache(cache_dir: str = None, models: list = None):
    """Download tiktoken models to local cache

    Args:
        cache_dir: Directory to store the cache files. If None, uses
            ``$TIKTOKEN_CACHE_DIR`` / ``$DATA_GYM_CACHE_DIR`` when set, else
            ``~/.tiktoken_cache`` (never tiktoken's own temp-dir fallback,
            which is cleared on reboot on many hosts).

    Returns:
        Tuple of (success_count, failed_models)
    """
    from lightrag.tiktoken_cache import (
        configured_tiktoken_cache_dir,
        default_tiktoken_cache_dir,
    )

    # tiktoken resolves its cache directory from the environment on every
    # cache lookup, so exporting it here steers the downloads below.
    using_default = False
    if cache_dir is not None:
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        print(f"Using specified cache directory: {cache_dir}")
    elif configured_tiktoken_cache_dir():
        cache_dir = configured_tiktoken_cache_dir()
        print(f"Using cache directory from environment: {cache_dir}")
    else:
        cache_dir = default_tiktoken_cache_dir()
        if cache_dir is None:
            print(
                "Error: cannot resolve the home directory for the default "
                "~/.tiktoken_cache; pass --cache-dir or set TIKTOKEN_CACHE_DIR."
            )
            sys.exit(1)
        using_default = True
        print(f"Using default cache directory: {cache_dir}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir

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
    if using_default:
        print(
            "\nLightRAG reads this directory automatically while TIKTOKEN_CACHE_DIR"
            "\nis unset. Other tiktoken users need:"
            f"\n  export TIKTOKEN_CACHE_DIR={cache_dir}"
        )
    print("\nFor offline deployment:")
    print("  1. Copy directory to offline server:")
    print(f"     tar -czf tiktoken_cache.tar.gz -C {cache_dir} .")
    print("     scp tiktoken_cache.tar.gz user@offline-server:/path/to/")
    print("")
    print("  2. On offline server, extract into the default location")
    print("     (used by LightRAG automatically):")
    print("     mkdir -p ~/.tiktoken_cache")
    print("     tar -xzf tiktoken_cache.tar.gz -C ~/.tiktoken_cache")
    print("")
    print("  3. Or extract anywhere and point TIKTOKEN_CACHE_DIR at it:")
    print("     export TIKTOKEN_CACHE_DIR=/path/to/tiktoken_cache")

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
  # Download tiktoken cache to default location (~/.tiktoken_cache)
  lightrag-download-cache

  # Same, explicitly (equivalent to the default when no spaCy flag is given)
  lightrag-download-cache --tiktoken

  # --cache-dir/--models are tiktoken-specific: giving either is enough to
  # select the tiktoken download, no need to also pass --tiktoken
  lightrag-download-cache --cache-dir ./offline_cache/tiktoken
  lightrag-download-cache --models gpt-4o-mini gpt-4

  # Download the pinned spaCy model wheels for the native docx smart_heading
  # engine parameter (to ./spacy_models by default) — tiktoken is untouched
  lightrag-download-cache --spacy

  # Install the spaCy models straight into the current environment —
  # tiktoken is untouched
  lightrag-download-cache --spacy-install

  # Want both, without giving any directory overrides: combine --tiktoken
  # with --spacy/--spacy-install explicitly
  lightrag-download-cache --tiktoken --spacy-install

For more information, visit: https://github.com/HKUDS/LightRAG
        """,
    )

    parser.add_argument(
        "--tiktoken",
        action="store_true",
        help="Download the tiktoken cache. This is the default when neither "
        "--spacy nor --spacy-install is given; pass it explicitly to combine "
        "with a spaCy flag",
    )
    parser.add_argument(
        "--cache-dir",
        help="Cache directory path (default: $TIKTOKEN_CACHE_DIR if set, else "
        "~/.tiktoken_cache). Selects the "
        "tiktoken download on its own, same as --tiktoken",
        default=None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to download (default: common models). Selects "
        "the tiktoken download on its own, same as --tiktoken",
        default=None,
    )
    parser.add_argument(
        "--spacy",
        action="store_true",
        help="Download the pinned spaCy model wheels used by the native docx "
        "smart_heading engine parameter, independently of tiktoken",
    )
    parser.add_argument(
        "--spacy-dir",
        help="Directory for the spaCy model wheels (default: ./spacy_models). "
        "Selects the spaCy download on its own, same as --spacy",
        default=None,
    )
    parser.add_argument(
        "--spacy-install",
        action="store_true",
        help="Install the spaCy models into the current environment (with "
        "pip, or uv when the environment has no pip) instead of downloading "
        "wheels, independently of tiktoken",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s (LightRAG cache downloader)"
    )

    args = parser.parse_args()

    tiktoken_detail_given = args.cache_dir is not None or args.models is not None
    spacy_detail_given = args.spacy_dir is not None
    want_spacy = args.spacy or args.spacy_install or spacy_detail_given
    want_tiktoken = args.tiktoken or tiktoken_detail_given or not want_spacy

    print("=" * 70)
    print("LightRAG Offline Cache Downloader")
    print("=" * 70)

    try:
        success_count, failed_models = (
            download_tiktoken_cache(args.cache_dir, args.models)
            if want_tiktoken
            else (0, [])
        )

        if want_spacy:
            spacy_success, spacy_failed = download_spacy_models(
                args.spacy_dir, install=args.spacy_install
            )
            success_count += spacy_success
            failed_models.extend(spacy_failed)

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
