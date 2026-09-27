"""Frozen DINO-family global descriptors (torch.hub DINOv2; DINOv3-equivalent slot)."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# DINOv3 not available offline in this env; DINOv2 ViT-S/14 is the documented equivalent.
DEFAULT_HUB = "facebookresearch/dinov2"
DEFAULT_MODEL = "dinov2_vits14"
EMBED_DIM = 384


@dataclass
class EmbedderMeta:
    hub_repo: str
    model_name: str
    embed_dim: int
    image_size: int
    normalize: str
    device: str
    torch_version: str
    note: str


class DinoGlobalEmbedder:
    def __init__(self, image_size: int = 224, device: str | None = None):
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load(DEFAULT_HUB, DEFAULT_MODEL, pretrained=True, trust_repo=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def meta(self) -> EmbedderMeta:
        return EmbedderMeta(
            hub_repo=DEFAULT_HUB,
            model_name=DEFAULT_MODEL,
            embed_dim=EMBED_DIM,
            image_size=self.image_size,
            normalize="l2_on_output",
            device=str(self.device),
            torch_version=torch.__version__,
            note="Proposal DINOv3 slot: using frozen DINOv2 ViT-S/14 (384-d) via torch.hub.",
        )

    @torch.inference_mode()
    def embed_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        out = self.model(batch)
        if isinstance(out, dict):
            out = out.get("x_norm_clstoken", next(iter(out.values())))
        out = F.normalize(out, dim=-1)
        return out.cpu().numpy().astype(np.float32)

    def embed_rgb_arrays(self, rgbs: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        vecs = []
        for i in range(0, len(rgbs), batch_size):
            chunk = [Image.fromarray(r.astype(np.uint8)) for r in rgbs[i : i + batch_size]]
            vecs.append(self.embed_pil_batch(chunk))
        return np.vstack(vecs) if vecs else np.zeros((0, EMBED_DIM), np.float32)


def save_embedding_cache(
    path: Path,
    embeddings: np.ndarray,
    tile_ids: np.ndarray,
    meta: EmbedderMeta,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(".meta.json")
    payload = {"meta": asdict(meta), "extra": extra or {}}
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(path, embeddings=embeddings, tile_ids=tile_ids)


def load_embedding_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return data["embeddings"], data["tile_ids"], meta
