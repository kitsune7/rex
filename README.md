# rex

A wake word detection system with speech recognition capabilities.

## Setup

1. **Install dependencies** (uses uv for package management):
   ```bash
   uv sync
   ```

2. **First-time model setup**:
   The required openwakeword models (~15MB) will be automatically downloaded on first use. Alternatively, you can manually download them:
   ```bash
   uv run python -c "from openwakeword.utils import download_models; download_models()"
   ```

3. Run `brew install espeak-ng` to get text to speech working.

## Usage

### Test a pre-trained wake word model:
```bash
uv run wake_word test-pretrained wake_word_training_files/hey_recks.onnx
```

The command will start listening for the wake word. Speak clearly into your microphone to test detection.

Press `Ctrl+C` to stop listening.

### Options:
- `--threshold`: Detection sensitivity (0.0-1.0, default: 0.5)
- `--chunk-size`: Audio chunk size in samples (default: 1280)

## Project Structure

- `src/wake_word/` - Wake word detection implementation
  - `cli.py` - Command-line interface
  - `wake_word_listener.py` - Real-time wake word detection
  - `model_utils.py` - Model management and auto-download
- `wake_word_training_files/` - Custom trained wake word models

