"""MRT2 JAX streaming engine: model lifecycle, conditioning, and audio yield."""

from __future__ import annotations

import base64
import gc
import os
import threading
import time
from dataclasses import dataclass

import jax
import numpy as np
import soundfile as sf

from magenta_rt.jax.system import discretize_cfg

from .clip_encoder import CLIPStyleEncoder
from .config import (
    AVAILABLE_MODELS,
    FRAMES_PER_SECOND,
    SAMPLE_RATE,
    SESSION_BUFFER_SEC,
    SESSION_DIR,
)


def samples_to_pcm_b64(samples: np.ndarray) -> str:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


@dataclass
class StreamMetrics:
    total_frames: int = 0
    last_ms_per_step: float = 0.0
    last_chunk_sec: float = 0.0
    last_clip_ms: float = 0.0
    frames_per_chunk: int = FRAMES_PER_SECOND
    clip_per_chunk: bool = False  # video/screen: CLIP runs every chunk

    def status_line(self, prompt: str, realtime_target_ms: float = 40.0) -> str:
        total_sec = self.total_frames / FRAMES_PER_SECOND
        prompt_preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
        n = self.last_ms_per_step
        m = self.last_clip_ms
        frames = self.frames_per_chunk
        include_clip = self.clip_per_chunk and m > 0
        used_ms = frames * n + (m if include_clip else 0)
        budget_ms = realtime_target_ms * frames
        ok = used_ms <= budget_ms
        rt_ok = "OK" if ok else "SLOW"
        sign = "<" if ok else ">"

        clip_line = (
            f"\nclip embed: {m:.1f} ms"
            if m > 0
            else ""
        )
        clip_term = f" + {m:.1f}" if include_clip else ""
        budget_line = (
            f"\n{frames}×{n:.1f}{clip_term} = {used_ms:.0f} ms "
            f"{sign} {realtime_target_ms:.0f}×{frames} = {budget_ms:.0f} ms "
            f"(realtime {rt_ok})"
        )
        return (
            f"prompt: {prompt_preview!r}\n"
            f"stream time: {total_sec:.1f}s ({self.total_frames} frames)\n"
            f"music gen: {self.last_chunk_sec:.2f}s/chunk, "
            f"{n:.1f} ms/step"
            f"{clip_line}"
            f"{budget_line}"
        )


class MRT2StreamEngine:
    def __init__(
        self,
        model: str = "mrt2_small",
        checkpoint: str | None = None,
        temperature: float = 1.3,
        top_k: int = 40,
        cfg_musiccoca: float = 3.0,
        cfg_notes: float = 1.0,
        yield_frames: int = 25,
        clip_encoder: CLIPStyleEncoder | None = None,
    ):
        self.model_name = model
        self.checkpoint = checkpoint
        self.temperature = temperature
        self.top_k = top_k
        self.cfg_musiccoca = cfg_musiccoca
        self.cfg_notes = cfg_notes
        self.yield_frames = yield_frames
        self._clip_encoder = clip_encoder

        self._mrt = None
        self._state = None
        self._style = None
        self._style_tokens: list[int] | None = None
        self._block = None
        self._constants = None
        self._prompt = ""
        self._last_clip_ms = 0.0
        self._clip_recurring = False  # video/screen: CLIP runs every chunk
        self._gpu_initialized = False  # True once a model has been loaded
        self._session_chunks: list[np.ndarray] = []
        self._session_buffer_frames = 0
        self._lock = threading.Lock()
        self.metrics = StreamMetrics()

    def preload(self) -> None:
        """Load MRT2 JAX model into GPU memory at startup."""
        with self._lock:
            self._ensure_loaded()

    def model_status(self) -> dict:
        with self._lock:
            return {
                "model": self.model_name,
                "loaded": self._mrt is not None,
                "available": list(AVAILABLE_MODELS),
                "preallocate": os.environ.get(
                    "XLA_PYTHON_CLIENT_PREALLOCATE", "true"
                ).lower()
                != "false",
                "gpu_initialized": self._gpu_initialized,
            }

    def set_preallocate(self, enabled: bool) -> str:
        """Set XLA GPU preallocation. Only effective before the GPU backend
        is initialized (i.e. before the first model load)."""
        with self._lock:
            if self._gpu_initialized:
                cur = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
                return (
                    f"Cannot change preallocate: GPU already initialized (current {cur}). "
                    "Restart the server and set it before loading a model."
                )
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if enabled else "false"
            return f"preallocate = {'true' if enabled else 'false'} (applied from the next model load)"

    def _unload(self) -> None:
        """Drop MRT2 model and free device buffers (best effort)."""
        self._mrt = None
        self._state = None
        self._block = None
        self._constants = None
        self._style_tokens = None
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

    def set_model(self, model_name: str) -> str:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model {model_name!r}; choose from {sorted(AVAILABLE_MODELS)}"
            )
        with self._lock:
            if model_name == self.model_name and self._mrt is not None:
                return f"Model already loaded: {model_name}"
            if self._mrt is not None:
                print(f"Unloading MRT2 model: {self.model_name} ...")
                self._unload()
            self.model_name = model_name
            self._ensure_loaded()
            return f"Model loaded: {model_name}"

    def _ensure_loaded(self) -> None:
        if self._mrt is not None:
            return
        from magenta_rt import MagentaRT2Jax

        t0 = time.perf_counter()
        print(f"Loading MRT2 JAX model: {self.model_name} ...")
        self._mrt = MagentaRT2Jax(
            size=self.model_name,
            checkpoint=self.checkpoint,
            temperature=self.temperature,
            top_k=self.top_k,
            cfg_musiccoca=self.cfg_musiccoca,
            cfg_notes=self.cfg_notes,
        )
        self._gpu_initialized = True
        print(f"  model ready in {time.perf_counter() - t0:.1f}s")

    def set_yield_frames(self, frames: int) -> None:
        frames = max(1, int(frames))
        with self._lock:
            self.yield_frames = frames

    def _style_tokens_from_embedding(self) -> list[int]:
        mrt = self._mrt
        assert mrt is not None
        if self._style is None:
            tokens = [-1] * mrt._num_musiccoca_tokens
        else:
            tokens = mrt._style_model.tokenize(self._style).tolist()
        if len(tokens) < mrt._num_musiccoca_tokens:
            tokens = tokens + [-1] * (mrt._num_musiccoca_tokens - len(tokens))
        return tokens[: mrt._num_musiccoca_tokens]

    def _rebuild_conditioning(self) -> None:
        mrt = self._mrt
        assert mrt is not None
        self._style_tokens = self._style_tokens_from_embedding()
        cfgs = [
            discretize_cfg(self.cfg_musiccoca, 0.2, 40),
            discretize_cfg(self.cfg_notes, 0.2, 40),
            discretize_cfg(mrt.cfg_drums, 1.0, 8),
        ]
        self._block, self._constants = mrt._build_conditioning(
            self._style_tokens,
            cfgs=cfgs,
            temperature=self.temperature,
            top_k=self.top_k,
        )

    def _ensure_streaming_state(self) -> None:
        mrt = self._mrt
        assert mrt is not None
        if self._state is None:
            self._state = mrt._jit_init_state(mrt._params, {})
        if self._block is None or self._constants is None:
            self._rebuild_conditioning()

    def _steps_to_audio(self, step_outputs: list) -> np.ndarray:
        parts = [np.asarray(jax.device_get(out.values[0])) for out in step_outputs]
        samples = np.concatenate(parts, axis=0)
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        return samples.astype(np.float32) / 32768.0

    @staticmethod
    def _chunk_frames(samples: np.ndarray) -> int:
        if samples.ndim == 1:
            return len(samples) // 2
        return int(samples.shape[0])

    def _append_session_chunk(self, samples: np.ndarray) -> None:
        self._session_chunks.append(samples.copy())
        self._session_buffer_frames += self._chunk_frames(samples)
        max_frames = int(SESSION_BUFFER_SEC * SAMPLE_RATE)
        while self._session_chunks and self._session_buffer_frames > max_frames:
            removed = self._session_chunks.pop(0)
            self._session_buffer_frames -= self._chunk_frames(removed)

    def _clear_session_chunks(self) -> None:
        self._session_chunks = []
        self._session_buffer_frames = 0

    def update_sampling(
        self,
        temperature: float,
        top_k: int,
        cfg_musiccoca: float,
        cfg_notes: float,
    ) -> None:
        with self._lock:
            self.temperature = temperature
            self.top_k = top_k
            self.cfg_musiccoca = cfg_musiccoca
            self.cfg_notes = cfg_notes
            if self._mrt is not None:
                self._mrt.temperature = temperature
                self._mrt.top_k = top_k
                self._mrt.cfg_musiccoca = cfg_musiccoca
                self._mrt.cfg_notes = cfg_notes
            if self._mrt is not None and self._block is not None:
                self._rebuild_conditioning()

    def set_prompt(self, prompt: str, *, reset_state: bool = False) -> str:
        prompt = prompt.strip()
        with self._lock:
            self._ensure_loaded()
            self._prompt = prompt
            self._last_clip_ms = 0.0
            self._clip_recurring = False
            if prompt:
                t0 = time.perf_counter()
                self._style = self._mrt.embed_style(prompt, use_mapper=True)
                embed_ms = (time.perf_counter() - t0) * 1000
                msg = f"Prompt embedded in {embed_ms:.1f} ms"
            else:
                self._style = None
                msg = "Empty prompt → unconditional generation"
            self._rebuild_conditioning()
            if reset_state:
                self._state = None
                self._clear_session_chunks()
                self.metrics = StreamMetrics()
                msg += " (state reset)"
        return msg

    def apply_image_b64(
        self,
        image_b64: str,
        filename: str = "image",
        *,
        reset_state: bool = False,
        recurring: bool = False,
    ) -> str:
        if self._clip_encoder is None:
            raise RuntimeError("CLIP encoder not configured")
        with self._lock:
            self._ensure_loaded()
            t0 = time.perf_counter()
            style = self._clip_encoder.embed_from_b64(image_b64)
            embed_ms = (time.perf_counter() - t0) * 1000
            self._last_clip_ms = embed_ms
            self._clip_recurring = recurring
            self._prompt = f"[image] {filename}"
            self._style = style
            self._rebuild_conditioning()
            norm = float(np.linalg.norm(style))
            if reset_state:
                self._state = None
                self._clear_session_chunks()
                self.metrics = StreamMetrics()
            msg = f"Image embedded (CLIP) in {embed_ms:.1f} ms (norm={norm:.4f})"
            if reset_state:
                msg += " (state reset)"
        return msg

    def reset_state(self) -> str:
        with self._lock:
            self._state = None
            self._block = None
            self._constants = None
            self._clear_session_chunks()
            self.metrics = StreamMetrics()
        return "State reset. Next Play starts from a fresh context."

    def export_session_wav(self) -> str | None:
        with self._lock:
            if not self._session_chunks:
                return None
            samples = np.concatenate(self._session_chunks, axis=0)
            out_path = SESSION_DIR / "session_export.wav"
            tmp_path = out_path.with_name(f"{out_path.stem}.part.wav")
            sf.write(str(tmp_path), samples, SAMPLE_RATE)
            os.replace(str(tmp_path), str(out_path))
            return str(out_path)

    def stream_yield_batch(self) -> tuple[np.ndarray, StreamMetrics, str]:
        with self._lock:
            self._ensure_loaded()
            self._ensure_streaming_state()
            mrt = self._mrt
            assert mrt is not None
            assert self._block is not None
            assert self._constants is not None

            t0 = time.perf_counter()
            step_outputs = []
            for _ in range(self.yield_frames):
                step_output, self._state, _ = mrt._jit_streaming_step(
                    mrt._params, self._block, self._constants, self._state
                )
                step_outputs.append(step_output)

            samples = self._steps_to_audio(step_outputs)
            self._append_session_chunk(samples)
            elapsed = time.perf_counter() - t0
            self.metrics.total_frames += self.yield_frames
            self.metrics.last_ms_per_step = (elapsed / self.yield_frames) * 1000
            self.metrics.last_chunk_sec = elapsed
            prompt = self._prompt
            metrics = StreamMetrics(
                total_frames=self.metrics.total_frames,
                last_ms_per_step=self.metrics.last_ms_per_step,
                last_chunk_sec=self.metrics.last_chunk_sec,
                last_clip_ms=self._last_clip_ms,
                frames_per_chunk=self.yield_frames,
                clip_per_chunk=self._clip_recurring,
            )
        return samples, metrics, prompt
