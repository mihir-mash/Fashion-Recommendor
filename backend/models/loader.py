import logging
from config import Settings
from typing import Any

logger = logging.getLogger("backend.models.loader")
settings = Settings()


def _stub_warn(name: str):
    logger.warning(f"STUB MODEL ACTIVE: replace with real model at models/{name}")


def load_image_model(path: str = None) -> Any:
    path = path or settings.IMAGE_MODEL_PATH
    try:
        from .image_model import load_image_model as _loader
        return _loader(path)
    except Exception as e:
        logger.warning(f"Failed to load image model from {path}: {e}", exc_info=True)
        _stub_warn('image_model')

        class StubImageModel:
            def embed_image(self, image):
                import numpy as np
                # deterministic fake embedding
                arr = np.random.RandomState(0).randn(512).astype("float32")
                return arr

        return StubImageModel()


def load_text_model(path: str = None) -> Any:
    path = path or settings.TEXT_MODEL_PATH
    try:
        from .text_model import load_text_model as _loader
        return _loader(path)
    except Exception as e:
        logger.warning(f"Failed to load text model from {path}: {e}", exc_info=True)
        _stub_warn('text_model')

        class StubTextModel:
            def embed_text(self, text: str):
                import numpy as np
                seed = abs(hash(text)) % (2**32 - 1)
                return np.random.RandomState(seed).randn(512).astype("float32")

        return StubTextModel()


def load_outfit_model(path: str = None) -> Any:
    path = path or settings.OUTFIT_MODEL_PATH
    try:
        from .outfit_model import load_outfit_model as _loader
        return _loader(path)
    except Exception as e:
        logger.warning(f"Failed to load outfit model from {path}: {e}", exc_info=True)
        _stub_warn('outfit_model')

        class StubOutfitModel:
            def predict_matches(self, embedding, target_type: str, k: int = 6):
                import numpy as np
                # deterministic fake: return k ids with descending scores
                ids = [str(i) for i in range(k)]
                scores = [float(1.0 - i * (0.1)) for i in range(k)]
                return list(zip(ids, scores))

        return StubOutfitModel()
