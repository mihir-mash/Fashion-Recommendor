from typing import Any
import numpy as np
import logging

logger = logging.getLogger("backend.models.outfit_model")


def load_outfit_model(path: str) -> Any:
    logger.info(f"Attempting to load outfit model from {path}")

    class SimpleOutfitModel:
        def predict_matches(self, embedding: np.ndarray, target_type: str, k: int = 6):
            # stub: return dummy ids based on embedding sum
            s = int(abs(embedding.sum()))
            ids = [str((s + i) % 1000) for i in range(k)]
            scores = [float(1.0 - i * 0.05) for i in range(k)]
            return list(zip(ids, scores))

    return SimpleOutfitModel()
