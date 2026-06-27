"""Shared configuration: env defaults, constants, and filesystem paths."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# magenta_rt reads MAGENTA_HOME on import; set a default before it is needed.
os.environ.setdefault("MAGENTA_HOME", "/mnt/Magenta")

# Audio / streaming
SAMPLE_RATE = 48000
FRAMES_PER_SECOND = 25
SESSION_BUFFER_SEC = 60  # session buffer for export/memory (keep only the last N seconds)

# Models
AVAILABLE_MODELS = ("mrt2_small", "mrt2_base")

# CLIP style encoder
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
CLIP_EMBED_DIM = 512
STYLE_EMBED_DIM = 768
DEFAULT_PROJECTION = "./weight/clip_proj.pt"

# Filesystem paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_DIR / "static"
FAVICON_PATH = STATIC_DIR / "favicon.ico"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

SESSION_DIR = Path(tempfile.gettempdir()) / "mrt2_web"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
