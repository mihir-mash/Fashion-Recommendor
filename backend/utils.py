from PIL import Image
import io
import numpy as np
from typing import List
import logging

logger = logging.getLogger("backend.utils")


def image_to_pil(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content)).convert('RGB')


def normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype='float32')
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / (n + 1e-9)


def color_histogram_score(img: Image.Image, target_color: str) -> float:
    # very simple heuristic: match average color name approximation
    arr = np.array(img.resize((16, 16))).astype('float32')
    avg = arr.mean(axis=(0, 1))
    # rudimentary mapping for a few colors
    mapping = {
        'red': np.array([200, 40, 40]),
        'blue': np.array([50, 80, 200]),
        'green': np.array([60, 160, 60]),
        'black': np.array([20, 20, 20]),
        'white': np.array([230, 230, 230]),
    }
    ref = mapping.get(target_color.lower()) if target_color else None
    if ref is None:
        return 0.0
    dist = np.linalg.norm(avg - ref)
    score = max(0.0, 1.0 - dist / 300.0)
    return float(score)


def mk_explanation(meta_row: dict, score: float) -> List[str]:
    explain = []
    if not meta_row:
        return explain
    if 'masterCategory' in meta_row and meta_row.get('masterCategory'):
        explain.append(f"Same category ({meta_row.get('masterCategory')})")
    if 'baseColour' in meta_row and meta_row.get('baseColour'):
        explain.append(f"Color match: {meta_row.get('baseColour')}")
    if len(explain) == 0:
        explain.append("Similar style")
    return explain[:3]
