from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from weather import fetch_weather, map_weather_to_season
from embeddings import load_embeddings, load_meta
from indexer import search_index, load_index
import numpy as np

router = APIRouter()
logger = logging.getLogger("backend.routes.weather")

_index = load_index()
_embeddings = load_embeddings()
_meta = load_meta()


@router.get("/weather")
def recommend_weather(lat: Optional[float] = None, lon: Optional[float] = None, city: Optional[str] = None, k: int = 6):
    try:
        w = fetch_weather(lat=lat, lon=lon, city=city)
        mapped = map_weather_to_season(w)
        # filter meta by categories
        cats = mapped.get('categories', [])
        df = _meta
        if 'masterCategory' in df.columns:
            candidates = df[df['masterCategory'].isin(cats)]
        else:
            candidates = df
        if len(candidates) == 0:
            # fallback random sample
            sample = df.sample(min(k, len(df))) if len(df) > 0 else []
            results = []
            for _, row in (sample.iterrows() if hasattr(sample, 'iterrows') else []):
                results.append({
                    'id': str(row.get('id')),
                    'product_display_name': row.get('product_display_name'),
                    'image_url': row.get('image_url'),
                })
            return {"location": w.get('name', {}), "season": mapped['season'], "results": results}
        # pick top-k by popularity if available
        if 'popularity' in candidates.columns:
            candidates = candidates.sort_values('popularity', ascending=False).head(k)
        else:
            candidates = candidates.head(k)
        results = []
        for _, row in candidates.iterrows():
            results.append({
                'id': str(row.get('id')),
                'product_display_name': row.get('product_display_name'),
                'image_url': row.get('image_url'),
            })
        return {"location": w.get('name', {}), "season": mapped['season'], "results": results}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail={"error": "weather_failure", "details": str(e)})
