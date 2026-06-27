"""CLIP image encoder + trained projection to MusicCoCa-style embeddings."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import CLIP_EMBED_DIM, CLIP_MODEL_NAME, STYLE_EMBED_DIM


class CLIPWithProjection(nn.Module):
    """CLIP image encoder (frozen) + trained MLP projection → 768d style."""

    def __init__(self, clip_model_name: str = CLIP_MODEL_NAME):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(CLIP_EMBED_DIM, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, STYLE_EMBED_DIM),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=pixel_values)
            if not isinstance(image_features, torch.Tensor):
                vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
                pooled = vision_outputs.pooler_output
                if pooled is None:
                    pooled = vision_outputs.last_hidden_state[:, 0, :]
                image_features = self.clip.visual_projection(pooled)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        projected = self.projection(image_features)
        return projected / projected.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class CLIPStyleEncoder:
    """Load CLIP projection at startup; embed images on CPU."""

    def __init__(
        self,
        projection_path: Path,
        device: str = "cpu",
        clip_model_name: str = CLIP_MODEL_NAME,
        l2_normalize: bool = True,
    ):
        self.device = device
        self.l2_normalize = l2_normalize
        if not projection_path.is_file():
            raise FileNotFoundError(f"Projection checkpoint not found: {projection_path}")

        t0 = time.perf_counter()
        print(f"Loading CLIP projection from {projection_path} (device={device}) ...")
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = CLIPWithProjection(clip_model_name)
        state = torch.load(projection_path, map_location=device)
        self.model.projection.load_state_dict(state)
        self.model = self.model.to(device)
        self.model.eval()
        print(f"  CLIP ready in {time.perf_counter() - t0:.1f}s")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        projected = self.model(pixel_values).squeeze(0).cpu().numpy().astype(np.float32)
        if self.l2_normalize:
            norm = np.linalg.norm(projected)
            if norm > 0:
                projected = projected / norm
        return projected

    def embed_from_b64(self, image_b64: str) -> np.ndarray:
        raw = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw))
        return self.embed_image(image)
