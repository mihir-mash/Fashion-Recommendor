from typing import Any
import numpy as np
import logging

logger = logging.getLogger("backend.models.outfit_model")


def load_outfit_model(path: str) -> Any:
    """Load a pre-trained outfit model from a pickle file.
    
    The model should have a predict_matches method that takes:
    - embedding: np.ndarray
    - target_type: str (optional)
    - k: int (number of matches to return)
    And returns a list of tuples: [(item_id, score), ...]
    """
    from pathlib import Path
    import joblib
    
    logger.info(f"Attempting to load outfit model from {path}")
    
    p = Path(path)
    if not p.is_absolute():
        # If path already starts with 'models/', resolve from backend root
        # Otherwise resolve relative to this file's directory
        if str(path).startswith('models/'):
            # Path is like 'models/outfit_model.pkl', resolve from backend root
            p = (Path(__file__).parent.parent / path).resolve()
        else:
            # Path is relative to models directory
            p = (Path(__file__).parent / path).resolve()
    
    if not p.exists():
        logger.warning(f"Outfit model file not found at {p}, using stub model")
        class SimpleOutfitModel:
            def predict_matches(self, embedding: np.ndarray, target_type: str, k: int = 6):
                # stub: return dummy ids based on embedding sum
                s = int(abs(embedding.sum()))
                ids = [str((s + i) % 1000) for i in range(k)]
                scores = [float(1.0 - i * 0.05) for i in range(k)]
                return list(zip(ids, scores))
        return SimpleOutfitModel()
    
    try:
        # Load the model dictionary
        model_dict = joblib.load(p)
        logger.info(f"Successfully loaded outfit model from {p}")
        
        # Check if it's a dictionary (new format) or a class instance (old format)
        if isinstance(model_dict, dict):
            # New format: dictionary with embeddings, categories, etc.
            logger.info("Loading dictionary-based outfit model...")
            
            embeddings = model_dict['embeddings']
            categories = model_dict['categories']
            item_ids = model_dict['item_ids']
            complementary_categories = model_dict.get('complementary_categories', {})
            
            # Load FAISS index if available
            faiss_index = None
            if model_dict.get('has_faiss') and 'faiss_index_path' in model_dict:
                try:
                    import faiss
                    faiss_index = faiss.read_index(model_dict['faiss_index_path'])
                    logger.info("Loaded FAISS index for outfit model")
                except Exception as e:
                    logger.warning(f"Could not load FAISS index: {e}")
            
            # Create model wrapper
            class LoadedOutfitModel:
                def __init__(self, embeddings, categories, item_ids, faiss_index, complementary_categories):
                    self.embeddings = embeddings
                    self.categories = categories
                    self.item_ids = item_ids
                    self.faiss_index = faiss_index
                    self.complementary_categories = complementary_categories
                
                def predict_matches(self, embedding: np.ndarray, target_type: str = None, k: int = 6):
                    """Predict matching items for an outfit."""
                    # Normalize embedding
                    embedding = embedding.reshape(1, -1)
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                    embedding = embedding.astype('float32')
                    
                    # Search for similar items
                    if self.faiss_index is not None:
                        try:
                            import faiss
                            faiss.normalize_L2(embedding)
                            similarities, indices = self.faiss_index.search(embedding, min(k * 10, len(self.embeddings)))
                            similarities = similarities[0]
                            indices = indices[0]
                        except Exception:
                            # Fallback to numpy
                            similarities = (self.embeddings @ embedding.T).flatten()
                            indices = np.argsort(-similarities)[:min(k * 10, len(self.embeddings))]
                            similarities = similarities[indices]
                    else:
                        # Use numpy cosine similarity
                        similarities = (self.embeddings @ embedding.T).flatten()
                        indices = np.argsort(-similarities)[:min(k * 10, len(self.embeddings))]
                        similarities = similarities[indices]
                    
                    # Filter by target_type if provided
                    results = []
                    for sim, idx in zip(similarities, indices):
                        item_id = str(self.item_ids[idx])
                        category = self.categories[idx]
                        
                        # If target_type specified, only include matching categories
                        if target_type:
                            complementary = self.complementary_categories.get(target_type.lower(), [])
                            if category.lower() not in [c.lower() for c in complementary]:
                                continue
                        
                        results.append((item_id, float(sim)))
                        if len(results) >= k:
                            break
                    
                    return results
            
            return LoadedOutfitModel(embeddings, categories, item_ids, faiss_index, complementary_categories)
        
        elif hasattr(model_dict, 'predict_matches'):
            # Old format: class instance
            logger.info("Loaded class-based outfit model")
            return model_dict
        else:
            raise ValueError("Unknown model format")
            
    except Exception as e:
        logger.warning(f"Failed to load outfit model from {p}: {e}. Using stub model.")
        # Fallback to stub
        class SimpleOutfitModel:
            def predict_matches(self, embedding: np.ndarray, target_type: str = None, k: int = 6):
                # stub: return dummy ids based on embedding sum
                s = int(abs(embedding.sum()))
                ids = [str((s + i) % 1000) for i in range(k)]
                scores = [float(1.0 - i * 0.05) for i in range(k)]
                return list(zip(ids, scores))
        return SimpleOutfitModel()
