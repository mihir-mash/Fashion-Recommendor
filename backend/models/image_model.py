from typing import Any
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("backend.models.image_model")


def load_image_model(path: str) -> Any:
    # Placeholder loader: teammates should replace with actual model loader
    logger.info(f"Attempting to load image model from {path}")

    class SimpleImageModel:
        def embed_image(self, image: Image.Image) -> np.ndarray:
            # deterministic hash-based embedding for stub purposes
            arr = np.asarray(image.resize((32, 32)).convert("RGB"), dtype=np.uint8)
            flat = arr.mean(axis=(0, 1)).astype("float32")
            vec = np.repeat(flat, 170)[:512].astype("float32")
            # normalize
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            return vec

    return SimpleImageModel()
