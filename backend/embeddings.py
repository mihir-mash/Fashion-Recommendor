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
    """Load metadata CSV into a DataFrame and build id->index mapping.

    - `meta_path` can be provided or taken from `settings.META_PATH`.
    - If image files are present next to the CSV in an `images/` folder,
      `image_url` will be set to `/static/images/<id>.<ext>` so the frontend
      can fetch them from the static mount.
    """
    global _meta_df, _id_to_idx
    import os
    from pathlib import Path

    meta_path = meta_path or settings.META_PATH
    # resolve relative paths relative to the backend folder
    p = Path(meta_path)
    if not p.is_absolute():
        p = (Path(__file__).parent / meta_path).resolve()

    try:
        # Try to read CSV with error handling for malformed lines
        # on_bad_lines='skip' will skip lines that can't be parsed
        try:
            df = pd.read_csv(p, on_bad_lines='skip', encoding='utf-8')
        except TypeError:
            # Fallback for older pandas versions that don't have on_bad_lines
            df = pd.read_csv(p, error_bad_lines=False, warn_bad_lines=True, encoding='utf-8')

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
        images_dir = os.path.join(p.parent, 'images')
        def _find_image_for_id(row_id: str) -> str:
            # prefer HTTP-accessible URL under /static/images/ if the file exists on disk
            for ext in ('.jpg', '.jpeg', '.png'):
                cand = os.path.join(images_dir, f"{row_id}{ext}")
                if os.path.exists(cand):
                    # return the URL path the frontend can fetch (backend mounts this directory at /static/images)
                    return f"/static/images/{row_id}{ext}"
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
    except Exception as e:
        logger.warning(f"Could not load meta at {p}; creating empty meta. Error: {type(e).__name__}: {e}")
        _meta_df = pd.DataFrame(columns=['id', 'product_display_name', 'image_url', 'masterCategory', 'baseColour', 'popularity'])
        _id_to_idx = {}
        return _meta_df


def load_embeddings(emb_path: str = None) -> np.ndarray:
    global _embeddings
    from pathlib import Path
    
    emb_path = emb_path or settings.EMBEDDINGS_PATH
    # resolve relative paths relative to the backend folder
    p = Path(emb_path)
    if not p.is_absolute():
        p = (Path(__file__).parent / emb_path).resolve()
    
    try:
        _embeddings = np.load(p)
        logger.info(f"Loaded embeddings from {p}")
        return _embeddings
    except Exception as e:
        logger.warning(f"Could not load embeddings at {p}; creating random stub embeddings. Error: {type(e).__name__}: {e}")
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
