"""
Text-to-Speech synthesis using Microsoft VibeVoice-Realtime-0.5B.

Routes all audio through AudioManager to avoid race conditions.
"""

from __future__ import annotations

import copy
import logging
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from audio.manager import AudioManager

SAMPLE_RATE = 24_000
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_VOICE_PRESET = Path(__file__).resolve().parents[2] / "models" / "vibevoice_voices" / "en-Carter_man.pt"
DEFAULT_CFG_SCALE = 1.5
DEFAULT_INFERENCE_STEPS = 5


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def _silence_load_noise():
    """Suppress the expected-but-noisy transformers warnings during model load.

    VibeVoice-Realtime ships with the encoder half of the acoustic tokenizer
    absent (only the decoder is used at inference), and its tokenizer class is
    reported as Qwen2Tokenizer but loaded via VibeVoiceTextTokenizerFast. Both
    are benign, so we quiet them only around the from_pretrained calls.
    """
    from transformers.utils import logging as hf_logging

    loggers = [
        hf_logging.get_logger("transformers.modeling_utils"),
        hf_logging.get_logger("transformers.tokenization_utils_base"),
    ]
    prior_levels = [lg.level for lg in loggers]
    prior_verbosity = hf_logging.get_verbosity()
    try:
        for lg in loggers:
            lg.setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            yield
    finally:
        hf_logging.set_verbosity(prior_verbosity)
        for lg, level in zip(loggers, prior_levels):
            lg.setLevel(level)


class VibeVoiceVoice:
    """
    Wraps VibeVoice-Realtime for streaming single-speaker TTS.

    Loads the model + processor once, caches a voice preset, and exposes
    ``synthesize(text)`` as a generator of float32 numpy audio chunks at
    ``self.sample_rate``. Playback interruption is handled by the caller
    via AudioManager.
    """

    def __init__(
        self,
        voice_preset: str | Path = DEFAULT_VOICE_PRESET,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        inference_steps: int = DEFAULT_INFERENCE_STEPS,
    ):
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        self.sample_rate = SAMPLE_RATE
        self.cfg_scale = cfg_scale
        self.inference_steps = inference_steps

        self._device = device or _select_device()
        self._torch_device = torch.device(self._device)

        preset_path = Path(voice_preset)
        if not preset_path.exists():
            raise FileNotFoundError(
                f"VibeVoice voice preset not found: {preset_path}. "
                "Download one from https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model"
            )

        if self._device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
            device_map = None
        elif self._device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "sdpa"
            device_map = "cuda"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"
            device_map = "cpu"

        with _silence_load_noise():
            self._processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
            self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            if self._device == "mps":
                self._model.to("mps")

            self._model.eval()
            self._model.model.noise_scheduler = self._model.model.noise_scheduler.from_config(
                self._model.model.noise_scheduler.config,
                algorithm_type="sde-dpmsolver++",
                beta_schedule="squaredcos_cap_v2",
            )
            self._model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            self._prefilled_outputs = torch.load(
                str(preset_path),
                map_location=self._torch_device,
                weights_only=False,
            )

    def _prepare_inputs(self, text: str) -> dict[str, Any]:
        processed = self._processor.process_input_with_cached_prompt(
            text=text.strip(),
            cached_prompt=self._prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            key: value.to(self._torch_device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 numpy audio chunks for the given text."""
        from vibevoice.modular.streamer import AudioStreamer

        if not text or not text.strip():
            return

        inputs = self._prepare_inputs(text.replace("’", "'").replace("“", '"').replace("”", '"'))
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list[BaseException] = []
        stop_event = threading.Event()

        def _run() -> None:
            try:
                self._model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self._processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    show_progress_bar=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(self._prefilled_outputs),
                )
            except BaseException as exc:  # noqa: BLE001 — re-raised in main thread
                errors.append(exc)
                audio_streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        try:
            for chunk in audio_streamer.get_stream(0):
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)

                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)

                peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                if peak > 1.0:
                    chunk = chunk / peak

                yield chunk.astype(np.float32, copy=False)
        finally:
            stop_event.set()
            audio_streamer.end()
            thread.join()
            if errors:
                raise errors[0]


def load_voice(
    voice_preset: str | Path = DEFAULT_VOICE_PRESET,
    model_path: str = DEFAULT_MODEL_PATH,
) -> VibeVoiceVoice:
    return VibeVoiceVoice(voice_preset=voice_preset, model_path=model_path)


def speak_text(
    text: str,
    voice_obj: VibeVoiceVoice,
    audio_manager: "AudioManager",
    interrupt_check: Callable[[], bool] | None = None,
) -> bool:
    """
    Speak text using TTS, with optional interruption support.

    Routes audio through AudioManager for coordinated playback.

    Args:
        text: Text to speak
        voice_obj: VibeVoiceVoice instance
        audio_manager: AudioManager instance for audio output
        interrupt_check: Optional callable returning True to interrupt

    Returns:
        True if interrupted, False if completed normally
    """
    for chunk in voice_obj.synthesize(text):
        if interrupt_check is not None and interrupt_check():
            audio_manager.stop_current()
            return True

        was_interrupted = audio_manager.queue_audio_blocking(
            chunk, voice_obj.sample_rate, interrupt_check
        )
        if was_interrupted:
            return True

    return False
