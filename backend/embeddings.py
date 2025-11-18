import numpy as np
import pandas as pd
import logging
from config import Settings

logger = logging.getLogger("backend.embeddings")
settings = Settings()

_meta_df = None
_embeddings = None
_id_to_idx = None


def load_meta(meta_path: str = None) -> pd.DataFrame:
    global _meta_df, _id_to_idx
    meta_path = meta_path or settings.META_PATH
    try:
        df = pd.read_csv(meta_path)
        _meta_df = df
        if 'id' in df.columns:
            _id_to_idx = {str(r['id']): i for i, r in df.reset_index().iterrows()}
        return df
    except Exception:
        logger.warning(f"Could not load meta at {meta_path}; creating empty meta")
        _meta_df = pd.DataFrame(columns=['id', 'product_display_name', 'image_url', 'masterCategory', 'baseColour', 'popularity'])
        _id_to_idx = {}
        return _meta_df


def load_embeddings(emb_path: str = None) -> np.ndarray:
    global _embeddings
    emb_path = emb_path or settings.EMBEDDINGS_PATH
    try:
        _embeddings = np.load(emb_path)
        logger.info(f"Loaded embeddings from {emb_path}")
        return _embeddings
    except Exception:
        logger.warning(f"Could not load embeddings at {emb_path}; creating random stub embeddings")
        # create random embeddings consistent with meta length (or 1000)
        n = 1000
        if _meta_df is not None and len(_meta_df) > 0:
            n = len(_meta_df)
        _embeddings = np.random.RandomState(0).randn(n, 512).astype('float32')
        return _embeddings


def get_embedding_by_id(item_id: str) -> np.ndarray:
    global _id_to_idx, _embeddings
    if _id_to_idx is None:
        load_meta()
    if _embeddings is None:
        load_embeddings()
    idx = _id_to_idx.get(str(item_id))
    if idx is None:
        raise KeyError(f"id {item_id} not found in meta")
    return _embeddings[idx]
