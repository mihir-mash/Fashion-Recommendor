from typing import Any
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("backend.models.image_model")


def load_image_model(path: str) -> Any:
    """Load a pre-trained image model from a pickle file.
    
    The model should have an embed_image method that takes:
    - image: PIL.Image.Image
    And returns a numpy array embedding.
    """
    from pathlib import Path
    import joblib
    
    logger.info(f"Attempting to load image model from {path}")
    
    p = Path(path)
    if not p.is_absolute():
        # If path already starts with 'models/', resolve from backend root
        # Otherwise resolve relative to this file's directory
        if str(path).startswith('models/'):
            # Path is like 'models/image_model.pkl', resolve from backend root
            p = (Path(__file__).parent.parent / path).resolve()
        else:
            # Path is relative to models directory
            p = (Path(__file__).parent / path).resolve()
    
    if not p.exists():
        logger.warning(f"Image model file not found at {p}, using stub model")
        class SimpleImageModel:
            def embed_image(self, image: Image.Image) -> np.ndarray:
                # deterministic hash-based embedding for stub purposes
                arr = np.asarray(image.resize((32, 32)).convert("RGB"), dtype=np.uint8)
                flat = arr.mean(axis=(0, 1)).astype("float32")
                # Match stored embeddings dimension (768)
                vec = np.repeat(flat, 256)[:768].astype("float32")
                # normalize
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                return vec
        return SimpleImageModel()
    
    try:
        # Try to load the model using safe unpickler
        from .pickle_helper import safe_load_pickle, extract_model_attributes
        
        try:
            model = safe_load_pickle(p)
            logger.info(f"Successfully loaded image model from {p}")
            
            # Verify it has the required method
            if hasattr(model, 'embed_image'):
                # Wrap the model to ensure dimensions match stored embeddings (768)
                class DimensionWrappedImageModel:
                    def __init__(self, original_model):
                        self._model = original_model
                        # Copy other attributes
                        for attr in dir(original_model):
                            if not attr.startswith('_') and attr != 'embed_image':
                                try:
                                    setattr(self, attr, getattr(original_model, attr))
                                except:
                                    pass
                    
                    def embed_image(self, image: Image.Image) -> np.ndarray:
                        emb = self._model.embed_image(image)
                        # Ensure dimension matches stored embeddings (768)
                        if len(emb) != 768:
                            # Pad or truncate to 768 dimensions
                            if len(emb) < 768:
                                emb = np.pad(emb, (0, 768 - len(emb)), mode='constant')
                            else:
                                emb = emb[:768]
                        return emb
                
                return DimensionWrappedImageModel(model)
            else:
                # Try to extract attributes and create a wrapper
                attrs = extract_model_attributes(model)
                if attrs:
                    logger.info(f"Extracted attributes from model: {list(attrs.keys())}")
                    # Create a wrapper that uses the extracted attributes
                    class WrappedImageModel:
                        def __init__(self, attrs_dict):
                            for k, v in attrs_dict.items():
                                setattr(self, k, v)
                        
                        def embed_image(self, image: Image.Image) -> np.ndarray:
                            # If we have a model attribute, try to use it
                            if hasattr(self, 'model'):
                                emb = self.model.embed_image(image)
                                # Ensure dimension matches stored embeddings (768)
                                if len(emb) != 768:
                                    # Pad or truncate to 768 dimensions
                                    if len(emb) < 768:
                                        emb = np.pad(emb, (0, 768 - len(emb)), mode='constant')
                                    else:
                                        emb = emb[:768]
                                return emb
                            # Otherwise use stub behavior with correct dimension
                            arr = np.asarray(image.resize((32, 32)).convert("RGB"), dtype=np.uint8)
                            flat = arr.mean(axis=(0, 1)).astype("float32")
                            # Match stored embeddings dimension (768)
                            vec = np.repeat(flat, 256)[:768].astype("float32")
                            vec = vec / (np.linalg.norm(vec) + 1e-9)
                            return vec
                    
                    return WrappedImageModel(attrs)
                else:
                    raise AttributeError("Model missing embed_image method and no extractable attributes")
        except Exception as e:
            logger.warning(f"Failed to load image model from {p}: {e}. Using stub model.")
            raise
    except Exception as e:
        logger.warning(f"Failed to load image model from {p}: {e}. Using stub model.")
        # Fallback to stub
        class SimpleImageModel:
            def embed_image(self, image: Image.Image) -> np.ndarray:
                # deterministic hash-based embedding for stub purposes
                arr = np.asarray(image.resize((32, 32)).convert("RGB"), dtype=np.uint8)
                flat = arr.mean(axis=(0, 1)).astype("float32")
                # Match stored embeddings dimension (768)
                vec = np.repeat(flat, 256)[:768].astype("float32")
                # normalize
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                return vec
        return SimpleImageModel()
