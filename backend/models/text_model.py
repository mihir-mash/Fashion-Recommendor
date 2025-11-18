from typing import Any
import numpy as np
import logging

logger = logging.getLogger("backend.models.text_model")


def load_text_model(path: str) -> Any:
    logger.info(f"Attempting to load text model from {path}")

    class SimpleTextModel:
        def embed_text(self, text: str) -> np.ndarray:
            seed = abs(hash(text)) % (2**32 - 1)
            return np.random.RandomState(seed).randn(512).astype("float32")

    return SimpleTextModel()
