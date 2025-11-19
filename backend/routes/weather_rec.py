from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any, Tuple
import logging
import time
from functools import lru_cache
import numpy as np
import pandas as pd

from weather import fetch_weather
from embeddings import load_embeddings, load_meta, get_embedding_by_id
from indexer import load_index, search_index  # search_index(index, query_vec, k) -> (scores, ids)
from models.loader import load_text_model  # optional; returns None if missing

router = APIRouter()
logger = logging.getLogger("backend.routes.weather")

# Supported cities (matching frontend)
SUPPORTED_CITIES = ["Hyderabad", "Shimla", "Goa", "Shillong", "Delhi"]

_index = load_index()            # may be None
_embeddings = load_embeddings()  # numpy array shaped (N,d) or None
_meta = load_meta()              # pandas DataFrame

# optional text model for season->embedding queries
_text_model = None
try:
    _text_model = load_text_model()  # should return None or object with embed_text(text)->np.ndarray
except Exception:
    logger.warning("text model not available for weather ranking; falling back to popularity/random")

# small cache for weather calls keyed by (lat,lon) or city - TTL implemented via lru_cache with manual time check
_WEATHER_TTL_SECONDS = 15 * 60  # 15 minutes
_weather_cache = {}  # key -> (timestamp, result)

# recommendation cache: key -> (timestamp, results)
_RECOMMENDATION_TTL_SECONDS = 30 * 60  # 30 minutes
_rec_cache = {}

# Pre-computed recommendations for all supported cities
_precomputed_recommendations = {}  # city_name -> recommendation dict

def _cache_weather(key: str, result: dict):
    _weather_cache[key] = (time.time(), result)

def _get_cached_weather(key: str):
    entry = _weather_cache.get(key)
    if not entry:
        return None
    ts, data = entry
    if time.time() - ts > _WEATHER_TTL_SECONDS:
        _weather_cache.pop(key, None)
        return None
    return data


def _get_cached_recommendation(key: str):
    entry = _rec_cache.get(key)
    if not entry:
        return None
    ts, data = entry
    if time.time() - ts > _RECOMMENDATION_TTL_SECONDS:
        _rec_cache.pop(key, None)
        return None
    return data


def _cache_recommendation(key: str, result: dict):
    _rec_cache[key] = (time.time(), result)

def _ensure_meta_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def _candidates_from_season(df: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
    if not cats:
        return df.copy()
    # normalize column name variations
    col = 'masterCategory' if 'masterCategory' in df.columns else ( 'category' if 'category' in df.columns else None)
    if col:
        return df[df[col].isin(cats)].copy()
    else:
        # can't filter by category if column missing; return full df
        return df.copy()


def _map_temp_to_dataset_season(temp_c: float) -> str:
    # follow the colab mapping: <=12 winter, <=20 autumn, <=28 spring, else summer
    try:
        if temp_c is None:
            return 'autumn'
        t = float(temp_c)
    except Exception:
        return 'autumn'
    if t <= 12:
        return 'winter'
    if t <= 20:
        return 'autumn'
    if t <= 28:
        return 'spring'
    return 'summer'

def _bruteforce_score_candidates(candidate_ids: List[int], query_vec: np.ndarray, embeddings: np.ndarray, topk: int):
    # candidate_ids are indices in embeddings array
    if embeddings is None:
        return [], []
    cand_embs = embeddings[candidate_ids]  # (M, d)
    # ensure normalized vectors
    cand_embs = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-10)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    sims = (cand_embs @ q).astype("float32")
    order = np.argsort(-sims)[:topk]
    top_idxs = [candidate_ids[int(i)] for i in order]
    top_scores = sims[order].tolist()
    return top_scores, top_idxs

# Default weather data for supported cities (used when API key is not available)
DEFAULT_CITY_WEATHER = {
    "Hyderabad": {"temp": 32, "condition": "Humid", "name": "Hyderabad"},
    "Shimla": {"temp": 15, "condition": "Cold", "name": "Shimla"},
    "Goa": {"temp": 30, "condition": "Sunny", "name": "Goa"},
    "Shillong": {"temp": 20, "condition": "Rainy", "name": "Shillong"},
    "Delhi": {"temp": 28, "condition": "Sunny", "name": "Delhi"},
}


def _get_weather_for_city(city_name: str) -> Dict:
    """Get weather for a city, using default data if API key is not available."""
    try:
        return fetch_weather(city=city_name)
    except RuntimeError as e:
        if "WEATHER_API_KEY" in str(e):
            # API key not configured, use default weather data
            if city_name in DEFAULT_CITY_WEATHER:
                default = DEFAULT_CITY_WEATHER[city_name]
                logger.info(f"Using default weather data for {city_name} (API key not configured)")
                return {
                    "name": default["name"],
                    "main": {"temp": default["temp"]},
                    "weather": [{"main": default["condition"]}]
                }
        raise


def _compute_recommendations_for_city(city_name: str, k: int = 6) -> Dict[str, Any]:
    """Internal function to compute recommendations for a given city."""
    try:
        w = _get_weather_for_city(city_name)
    except Exception as e:
        logger.exception(f"fetch_weather failed for {city_name}")
        # Return empty result if weather fetch fails
        return {
            "location": city_name,
            "season": "autumn",
            "total_candidates": 0,
            "results": []
        }
    
    # map to dataset season (winter/autumn/spring/summer) based on temperature
    try:
        temp = w.get('main', {}).get('temp')
        season = _map_temp_to_dataset_season(temp)
    except Exception as e:
        logger.exception("season mapping failed")
        season = "autumn"
    
    # get candidate dataframe
    df = _meta if isinstance(_meta, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        logger.warning("meta.csv empty or not loaded")
        return {"location": w.get("name", city_name), "season": season, "results": []}
    
    # Build season candidates from dataset 'season' column (supports comma-separated tags)
    if 'season_list' not in df.columns and 'season' in df.columns:
        df['season_list'] = df['season'].fillna("").apply(lambda s: [x.strip().lower() for x in str(s).split(",") if x.strip()])
    if 'season_list' in df.columns:
        season_candidates = df[df['season_list'].apply(lambda lst: season in lst if isinstance(lst, list) else False)].copy()
    else:
        # no season information; use entire df
        season_candidates = df.copy()
    
    total_candidates = len(season_candidates)
    if total_candidates == 0:
        # fallback: return top-k overall by popularity or random sample
        if 'popularity' in df.columns:
            top = df.sort_values('popularity', ascending=False).head(k)
        else:
            top = df.sample(min(k, len(df)))
        results = []
        for _, row in top.iterrows():
            # Try multiple possible image URL column names
            image_url = (
                row.get("image_url") or 
                row.get("imageURL") or 
                row.get("ImageURL") or
                row.get("image") or
                ""
            )
            # If image_url is a relative path, ensure it starts with /
            if image_url and not image_url.startswith(("http://", "https://", "/")):
                image_url = f"/static/images/{image_url}"
            results.append({
                "id": str(row.get("id", "")),
                "product_display_name": row.get("product_display_name", ""),
                "image_url": image_url
            })
        return {"location": w.get("name", city_name), "season": season, "total_candidates": total_candidates, "results": results}
    
    # Now apply the Colab-style Topwear pipeline: strict topwear -> relaxed substring -> coarse mapping -> random/sample fallback
    sc_df = season_candidates.copy()
    # normalize subCategory
    if 'subCategory' in sc_df.columns:
        sc_df['subCategory_norm'] = sc_df['subCategory'].fillna("").astype(str).str.strip().str.lower()
    elif 'sub_category' in sc_df.columns:
        sc_df['subCategory_norm'] = sc_df['sub_category'].fillna("").astype(str).str.strip().str.lower()
    else:
        sc_df['subCategory_norm'] = ""
    
    # strict Topwear
    candidates_topwear = sc_df[sc_df['subCategory_norm'] == 'topwear']
    if len(candidates_topwear) > 0:
        candidates = candidates_topwear.copy()
    else:
        # relaxed substring matches
        substrings = ['top', 't-shirt', 'tee', 'shirt', 'blouse', 'tshirt']
        mask = sc_df['subCategory_norm'].apply(lambda s: any(sub in s for sub in substrings))
        candidates_relaxed = sc_df[mask]
        if len(candidates_relaxed) > 0:
            candidates = candidates_relaxed.copy()
        else:
            # coarse mapping by season
            if season == 'winter':
                coarse_cats = ["coats", "jacket", "knitwear", "sweaters", "topwear"]
            elif season in ['autumn', 'spring']:
                coarse_cats = ["shirts", "tops", "hoodies", "long sleeve", "topwear"]
            else:
                coarse_cats = ["shorts", "tshirts", "dresses", "tops", "topwear"]
            candidates_coarse = sc_df[sc_df['subCategory_norm'].isin(coarse_cats)]
            if len(candidates_coarse) > 0:
                candidates = candidates_coarse.copy()
            else:
                if len(sc_df) == 0:
                    candidates = df.sample(min(k * 5, len(df))).reset_index(drop=True)
                else:
                    candidates = sc_df.sample(min(k * 5, len(sc_df)), random_state=42).reset_index(drop=True)
    
    # Ranking: if we have a text model and embeddings, use it; else fall back to popularity/random
    top_results = []
    try:
        if _text_model is not None and _index is not None and _embeddings is not None:
            prompt = f"{season} clothing"
            qvec = _text_model.embed_text(prompt)
            # candidate index list (positions in embeddings array)
            # Use the 'index' column which maps to embedding positions, not product IDs
            if 'index' in candidates.columns:
                candidate_idx_list = candidates['index'].astype(int).tolist()
            else:
                # Fallback to dataframe index (positional)
                candidate_idx_list = candidates.index.astype(int).tolist()
            
            # Filter to only include indices within embeddings array bounds
            max_emb_idx = len(_embeddings) - 1
            candidate_idx_list = [idx for idx in candidate_idx_list if 0 <= idx <= max_emb_idx]
            
            if not candidate_idx_list:
                # If no valid candidates, fall back to simple selection
                raise ValueError("No valid candidate indices")
            
            try:
                scores, ids = search_index(_index, qvec, k=k, embeddings=_embeddings, candidate_ids=candidate_idx_list)
            except (TypeError, IndexError, ValueError) as e:
                logger.warning(f"search_index failed: {e}, using bruteforce")
                scores, ids = _bruteforce_score_candidates(candidate_idx_list, qvec, _embeddings, k)
            
            for sc_score, idx in zip(scores, ids):
                try:
                    row = df.iloc[int(idx)]
                    # Try multiple possible image URL column names
                    image_url = (
                        row.get("image_url") or 
                        row.get("imageURL") or 
                        row.get("ImageURL") or
                        row.get("image") or
                        ""
                    )
                    # If image_url is empty but we have an id, try to construct it
                    if not image_url and 'id' in row:
                        item_id = str(row.get("id", ""))
                        if item_id:
                            # Try common image extensions
                            for ext in ['.jpg', '.jpeg', '.png']:
                                potential_path = f"/static/images/{item_id}{ext}"
                                image_url = potential_path
                                break
                    
                    # If image_url is a relative path, ensure it starts with /
                    if image_url and not image_url.startswith(("http://", "https://", "/")):
                        image_url = f"/static/images/{image_url}"
                    
                    top_results.append({
                        "id": str(row.get("id", str(idx))),
                        "product_display_name": row.get("product_display_name", ""),
                        "image_url": image_url,
                        "score": float(sc_score)
                    })
                except (IndexError, KeyError) as e:
                    logger.warning(f"Error accessing row at index {idx}: {e}")
                    continue
        else:
            if 'popularity' in candidates.columns:
                sel = candidates.sort_values('popularity', ascending=False).head(k)
            else:
                sel = candidates.head(k)
            for _, row in sel.iterrows():
                # Try multiple possible image URL column names
                image_url = (
                    row.get("image_url") or 
                    row.get("imageURL") or 
                    row.get("ImageURL") or
                    row.get("image") or
                    ""
                )
                # If image_url is a relative path, ensure it starts with /
                if image_url and not image_url.startswith(("http://", "https://", "/")):
                    image_url = f"/static/images/{image_url}"
                top_results.append({
                    "id": str(row.get("id", "")),
                    "product_display_name": row.get("product_display_name", ""),
                    "image_url": image_url
                })
    except Exception as e:
        logger.exception("error while ranking candidates; falling back to simple selection")
        sel = candidates.head(k)
        for _, row in sel.iterrows():
            top_results.append({
                "id": str(row.get("id", "")),
                "product_display_name": row.get("product_display_name", ""),
                "image_url": row.get("image_url", "")
            })
    
    return {
        "location": w.get("name", city_name),
        "season": season,
        "total_candidates": total_candidates,
        "results": top_results
    }


def _precompute_all_cities(k: int = 6):
    """Pre-compute recommendations for all supported cities."""
    logger.info(f"Pre-computing recommendations for {len(SUPPORTED_CITIES)} cities...")
    for city in SUPPORTED_CITIES:
        try:
            rec = _compute_recommendations_for_city(city, k)
            _precomputed_recommendations[city] = rec
            logger.info(f"Pre-computed recommendations for {city}: {len(rec.get('results', []))} items")
        except Exception as e:
            logger.exception(f"Failed to pre-compute recommendations for {city}")
            _precomputed_recommendations[city] = {
                "location": city,
                "season": "autumn",
                "total_candidates": 0,
                "results": []
            }
    logger.info("Finished pre-computing recommendations for all cities")


@router.get("/weather")
def recommend_weather(
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    city: Optional[str] = None,
    k: int = Query(6, ge=1, le=50)
) -> Dict[str, Any]:
    """
    Get weather-based recommendations.
    
    Accepts 'location' parameter (preferred) which should be one of the supported cities.
    Falls back to lat/lon or city for backward compatibility.
    """
    # Priority: location > city > lat/lon
    city_name = None
    if location:
        # Normalize location name (capitalize first letter)
        city_name = location.strip().capitalize()
        if city_name not in SUPPORTED_CITIES:
            # Try case-insensitive match
            city_name_lower = city_name.lower()
            for supported in SUPPORTED_CITIES:
                if supported.lower() == city_name_lower:
                    city_name = supported
                    break
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported location: {location}. Supported cities: {', '.join(SUPPORTED_CITIES)}"
                )
    elif city:
        city_name = city
    elif lat is not None and lon is not None:
        # Use lat/lon - fetch weather and compute on the fly
        try:
            w = fetch_weather(lat=lat, lon=lon)
            city_name = w.get("name", "Unknown")
        except Exception as e:
            logger.exception("fetch_weather failed")
            raise HTTPException(status_code=502, detail={"error": "weather_fetch_failed", "details": str(e)})
    else:
        raise HTTPException(status_code=400, detail="Provide 'location' (city name), 'city', or 'lat'/'lon'")
    
    # If it's a supported city and we have pre-computed results, return them
    if city_name in SUPPORTED_CITIES and city_name in _precomputed_recommendations:
        rec = _precomputed_recommendations[city_name].copy()
        # Adjust k if needed (return first k results)
        if len(rec.get("results", [])) > k:
            rec["results"] = rec["results"][:k]
        return rec
    
    # Otherwise, compute on the fly (for lat/lon or unsupported cities)
    return _compute_recommendations_for_city(city_name, k)