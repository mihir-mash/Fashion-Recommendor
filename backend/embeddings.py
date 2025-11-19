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
        import os

        df = pd.read_csv(meta_path)

        # Normalize common column names to the fields other modules expect
        # productDisplayName -> product_display_name
        if 'productDisplayName' in df.columns and 'product_display_name' not in df.columns:
            df['product_display_name'] = df['productDisplayName']

        # Ensure masterCategory and baseColour exist (some CSVs use different casing)
        if 'masterCategory' not in df.columns and 'master_category' in df.columns:
            df['masterCategory'] = df['master_category']
        if 'baseColour' not in df.columns and 'base_colour' in df.columns:
            df['baseColour'] = df['base_colour']

        # season column may exist; keep as-is

        # Build image_url if not present by probing images folder next to the CSV
        images_dir = os.path.join(os.path.dirname(meta_path), 'images')
        def _find_image_for_id(row_id: str) -> str:
            # try common extensions
            for ext in ('.jpg', '.jpeg', '.png'):
                p = os.path.join(images_dir, f"{row_id}{ext}")
                if os.path.exists(p):
                    # return a path relative to repo root so it's visible in logs; frontend may need hosting
                    return p.replace('\\', '/')
            return ''

        if 'image_url' not in df.columns:
            df['image_url'] = df['id'].apply(lambda x: _find_image_for_id(x)) if 'id' in df.columns else ''

        # Add an 'index' column that maps to embedding row index (positional)
        df = df.reset_index(drop=True)
        df['index'] = df.index.astype(int)

        _meta_df = df
        if 'id' in df.columns:
            _id_to_idx = {str(row['id']): int(idx) for idx, row in df.iterrows()}
        else:
            _id_to_idx = {}
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
