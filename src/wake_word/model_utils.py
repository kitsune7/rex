"""Utilities for managing wake word models."""

import sys
from pathlib import Path


def ensure_openwakeword_models() -> bool:
    """
    Ensure that required openwakeword resource models are available.

    Checks for the presence of required model files and downloads them
    if they're missing. This is a one-time setup that downloads ~15MB
    of model files.

    Returns:
        bool: True if models are available, False if download failed
    """
    try:
        import openwakeword

        openwakeword_dir = Path(openwakeword.__file__).parent
        resources_dir = openwakeword_dir / "resources" / "models"

        required_models = ["melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx"]

        models_exist = all((resources_dir / model).exists() for model in required_models)

        if not models_exist:
            print("📦 First-time setup: Downloading required openwakeword models...")
            print("   This is a one-time download (~15MB)")
            print()

            from openwakeword.utils import download_models

            download_models()

            print()
            print("✅ Model download complete!")
            print()

        return True

    except Exception as e:
        print(f"❌ Error downloading openwakeword models: {e}", file=sys.stderr)
        print("   Please try manually running:", file=sys.stderr)
        print(
            '   uv run python -c "from openwakeword.utils import download_models; download_models()"', file=sys.stderr
        )
        return False
