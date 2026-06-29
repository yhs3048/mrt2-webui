"""
MRT2 JAX realtime web UI (FastAPI + WebSocket).

No Gradio. Audio streams over WebSocket; browser plays gapless via Web Audio.
Drag-and-drop image/video, or Screen mode (1s display capture → CLIP embedding).

Usage:
  CUDA_VISIBLE_DEVICES=2 python webui.py --model mrt2_small
  CUDA_VISIBLE_DEVICES=2 python webui.py --host 0.0.0.0 --port 8000

Open http://<host>:8000 in your browser.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import uvicorn

from src.clip_encoder import CLIPStyleEncoder
from src.config import CLIP_MODEL_NAME, DEFAULT_PROJECTION
from src.engine import MRT2StreamEngine
from src.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRT2 JAX streaming web UI (FastAPI)")
    parser.add_argument("--model", default="mrt2_small")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--cfg-musiccoca", type=float, default=3.0)
    parser.add_argument("--cfg-notes", type=float, default=1.0)
    parser.add_argument("--yield-frames", type=int, default=25)
    parser.add_argument(
        "--projection",
        default=DEFAULT_PROJECTION,
        help="Trained CLIP projection checkpoint (train.py output)",
    )
    parser.add_argument("--clip-model", default=CLIP_MODEL_NAME)
    parser.add_argument(
        "--clip-device",
        default="cuda",
        help="Device for CLIP encoder: cpu, cuda, or cuda:N (default: cuda)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--preallocate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preallocate JAX GPU memory (XLA_PYTHON_CLIENT_PREALLOCATE).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        "true" if args.preallocate else "false"
    )

    clip_device = args.clip_device
    if clip_device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available for CLIP; falling back to CPU")
        clip_device = "cpu"

    clip_encoder = CLIPStyleEncoder(
        projection_path=Path(args.projection),
        device=clip_device,
        clip_model_name=args.clip_model,
    )
    engine = MRT2StreamEngine(
        model=args.model,
        checkpoint=args.checkpoint,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        yield_frames=args.yield_frames,
        clip_encoder=clip_encoder,
    )
    app = create_app(engine)

    print("\n=== MRT2 Web UI (FastAPI) ===")
    print("MRT2 model is loaded lazily (pick small/base in the UI or press Play).")
    print(f"Open: http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        print(f"Remote: http://<server-ip>:{args.port}")
    print("Docker: publish port, e.g. -p 8000:8000\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
