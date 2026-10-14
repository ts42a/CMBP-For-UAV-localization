"""Frozen DINO-family global descriptors (torch.hub DINOv2; DINOv3-equivalent slot)."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from integrity import sha256, digest
from PIL import Image
from torchvision import transforms

# DINOv3 not available offline in this env; DINOv2 ViT-S/14 is the documented equivalent.
DEFAULT_HUB = "facebookresearch/dinov2"
DEFAULT_MODEL = "dinov2_vits14"
EMBED_DIM = 384
PIN_PATH = Path(__file__).resolve().parents[2] / 'configs/dinov2_artifacts_v1.json'


def verify_model_artifacts():
    pin = json.loads(PIN_PATH.read_text(encoding='utf-8'))
    hub = Path(torch.hub.get_dir())
    repo = hub / pin['cache_directory']
    if not repo.is_dir():
        repo = hub / ('facebookresearch_dinov2_' + pin['revision'])
    if not repo.is_dir():
        raise RuntimeError('Pinned DINOv2 source absent; restore the verified offline bundle')
    actual = {p.relative_to(repo).as_posix() for p in repo.rglob('*.py')}
    if actual != set(pin['python_files']):
        raise RuntimeError('DINOv2 Python source file set differs from pin')
    for name, wanted in pin['python_files'].items():
        if sha256(repo/name) != wanted:
            raise RuntimeError(f'DINOv2 source digest mismatch: {name}')
    weights = hub/'checkpoints'/pin['weights_file']
    if not weights.is_file() or sha256(weights) != pin['weights_sha256']:
        raise RuntimeError('DINOv2 weights absent or digest mismatch')
    return repo, pin


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
    model_revision: str = ''
    weights_sha256: str = ''


class DinoGlobalEmbedder:
    def __init__(self, image_size: int = 224, device: str | None = None):
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        repo, self.pin = verify_model_artifacts()
        self.model = torch.hub.load(str(repo), DEFAULT_MODEL, source='local', pretrained=True)
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
            model_revision=self.pin['revision'], weights_sha256=self.pin['weights_sha256'],
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
    np.savez_compressed(path, embeddings=embeddings, tile_ids=tile_ids)
    if extra and 'rasters_sha256' in extra:
        payload['integrity'] = {'data_sha256':sha256(path), 'model_pin_sha256':sha256(PIN_PATH),
                                'rasters_sha256':extra['rasters_sha256']}
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_embedding_cache(path: Path, *, tiles=None, rasters=None) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    embs, ids = data['embeddings'], data['tile_ids']
    if embs.ndim != 2 or embs.shape[1] != EMBED_DIM or len(ids) != len(embs) or not np.isfinite(embs).all():
        raise ValueError(f'{path}: invalid embeddings')
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'{path}: duplicate tile IDs')
    m = meta['meta']
    if m['model_name'] != DEFAULT_MODEL or m['image_size'] != 224 or m['normalize'] != 'l2_on_output':
        raise ValueError(f'{path}: incompatible model/preprocessing metadata')
    if tiles is not None:
        if not np.array_equal(ids, [t.tile_id for t in tiles]):
            raise ValueError(f'{path}: ordered tile IDs differ')
        extra=meta.get('extra',{})
        if 'tiles' in extra:
            if extra['tiles'] != [asdict(t) for t in tiles]:
                raise ValueError(f'{path}: tile geometry differs')
        else:
            # Original single-raster extractor recorded georef instead of tile dictionaries.
            # Accept that explicit schema only on its original 1024/512 grid; the byte
            # registry below additionally binds this legacy artifact, not arbitrary caches.
            if extra.get('region') != '03' or extra.get('n_tiles') != len(tiles) or not tiles:
                raise ValueError(f'{path}: missing tile geometry')
            for t in tiles:
                if (t.raster_id != 0 or extra.get('georef') != [t.west,t.north,t.sx,t.sy]
                    or t.row0 % 512 or t.col0 % 512 or t.row1-t.row0 != 1024 or t.col1-t.col0 != 1024):
                    raise ValueError(f'{path}: legacy single-raster grid differs')
    if rasters is not None:
        lock_path = PIN_PATH.parent/'legacy_artifacts_v1.json'
        lock = meta.get('integrity') or json.loads(lock_path.read_text())['embedding_caches'].get(path.name)
        if not lock or sha256(path) != lock['data_sha256']:
            raise ValueError(f'{path}: cache identity is not in the verified inventory; rebuild/register explicitly')
        if 'metadata_sha256' in lock and sha256(path.with_suffix('.meta.json')) != lock['metadata_sha256']:
            raise ValueError(f'{path}: metadata identity differs')
        if lock['model_pin_sha256'] != sha256(PIN_PATH):
            raise ValueError(f'{path}: model identity differs')
        if meta.get('integrity'):
            pin = json.loads(PIN_PATH.read_text())
            if m.get('model_revision') != pin['revision'] or m.get('weights_sha256') != pin['weights_sha256']:
                raise ValueError(f'{path}: model revision/weights differ')
        actual = {r.filename:sha256(r.path) for r in rasters}
        if actual != lock['rasters_sha256']:
            raise ValueError(f'{path}: source raster identity differs')
    return data["embeddings"], data["tile_ids"], meta
