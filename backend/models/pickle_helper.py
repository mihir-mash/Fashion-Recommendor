"""Helper functions to load pickle files that may contain classes from __main__."""
import pickle
import joblib
import logging
from typing import Any, Dict

logger = logging.getLogger("backend.models.pickle_helper")


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that can handle missing class definitions."""
    
    def find_class(self, module, name):
        # If trying to load a class from __main__, try to return a stub
        if module == "__main__":
            logger.warning(f"Attempting to load class {name} from __main__, creating stub")
            # Return a generic object that can hold attributes
            class StubClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __setattr__(self, name, value):
                    object.__setattr__(self, name, value)
            return StubClass
        # For other modules, use default behavior
        return super().find_class(module, name)


def safe_load_pickle(file_path: str) -> Any:
    """Try to load a pickle file, handling missing class definitions."""
    try:
        # First try normal joblib load
        return joblib.load(file_path)
    except (AttributeError, ModuleNotFoundError) as e:
        if "__main__" in str(e) or "Can't get attribute" in str(e):
            logger.warning(f"Normal load failed for {file_path}, trying safe unpickler: {e}")
            try:
                # Try with custom unpickler
                with open(file_path, 'rb') as f:
                    unpickler = SafeUnpickler(f)
                    return unpickler.load()
            except Exception as e2:
                logger.error(f"Safe unpickler also failed for {file_path}: {e2}")
                raise
        else:
            raise


def extract_model_attributes(obj: Any) -> Dict[str, Any]:
    """Extract attributes from a model object into a dictionary."""
    attrs = {}
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            if not key.startswith('_') or key in ['_embeddings', '_item_ids', '_base_colours', '_article_types']:
                attrs[key] = value
    return attrs

