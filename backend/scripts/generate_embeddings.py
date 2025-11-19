"""Generate embeddings.npy from images using the image model loader.

Usage:
    python scripts/generate_embeddings.py --meta backend/data/archive/styles.csv --out backend/data/embeddings.npy

This will load the image model via models.loader.load_image_model() (stub if no real model present)
and embed each image listed in the CSV (by `id` -> backend/data/archive/images/<id>.(jpg|png)).
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ensure import path
from config import Settings
from models import loader

settings = Settings()


def find_image_path(images_dir: Path, row_id: str) -> Path | None:
    for ext in ('.jpg', '.jpeg', '.png'):
        p = images_dir / f"{row_id}{ext}"
        if p.exists():
            return p
    return None


def main(meta_path: str, out_path: str, images_dir: str | None = None):
    meta_path = Path(meta_path)
    if images_dir is None:
        images_dir = Path(meta_path.parent) / 'images'
    images_dir = Path(images_dir)
    out_path = Path(out_path)

    df = pd.read_csv(meta_path)
    # if product_display_name column exists, ok; ensure id exists
    if 'id' not in df.columns:
        raise SystemExit('meta CSV must contain an id column')

    model = loader.load_image_model()
    emb_list = []
    D = None

    for _, row in tqdm(df.reset_index(drop=True).iterrows(), total=len(df), desc='Embedding images'):
        rid = str(row['id'])
        img_path = find_image_path(images_dir, rid)
        if img_path is None:
            # fallback to placeholder image
            im = Image.new('RGB', (224, 224), color=(50, 50, 50))
        else:
            try:
                im = Image.open(img_path).convert('RGB')
            except Exception:
                im = Image.new('RGB', (224, 224), color=(50, 50, 50))
        vec = model.embed_image(im)
        vec = np.asarray(vec, dtype='float32')
        if D is None:
            D = vec.shape[0]
        if vec.shape[0] != D:
            # pad or truncate
            v = np.zeros((D,), dtype='float32')
            v[:min(D, vec.shape[0])] = vec[:min(D, vec.shape[0])]
            vec = v
        emb_list.append(vec)

    embs = np.stack(emb_list, axis=0).astype('float32')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), embs)
    print(f"Saved embeddings to {out_path} shape={embs.shape}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--meta', default=str(settings.META_PATH), help='path to meta CSV')
    p.add_argument('--out', default=str(settings.EMBEDDINGS_PATH), help='output .npy path')
    p.add_argument('--images', default=None, help='images directory (optional)')
    args = p.parse_args()
    main(args.meta, args.out, args.images)
