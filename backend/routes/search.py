from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi import Depends
from pydantic import BaseModel
from typing import Optional, List, Any
from PIL import Image
import io
import json
import numpy as np
import logging

from config import Settings
from models import loader
from indexer import load_index, search_index
from embeddings import load_embeddings, load_meta
from utils import image_to_pil, normalize_vec, color_histogram_score, mk_explanation

logger = logging.getLogger("backend.routes.search")
router = APIRouter()
settings = Settings()

# load resources
_image_model = loader.load_image_model()
_text_model = loader.load_text_model()
_index = load_index()
_embeddings = load_embeddings()
_meta = load_meta()


class TextSearchRequest(BaseModel):
    text: str
    k: Optional[int] = 6
    filters: Optional[dict] = None


@router.post("/image")
async def search_image(image: UploadFile = File(...), k: int = 6, filters: Optional[str] = Form(None)) -> Any:
    try:
        if _image_model is None:
            raise HTTPException(status_code=500, detail={"error": "image_model_not_loaded"})
        if _embeddings is None:
            raise HTTPException(status_code=500, detail={"error": "embeddings_not_loaded"})
        if _meta is None or len(_meta) == 0:
            raise HTTPException(status_code=500, detail={"error": "meta_not_loaded"})
            
        content = await image.read()
        pil = image_to_pil(content)
        q = _image_model.embed_image(pil)
        
        # Ensure embedding dimension matches stored embeddings (768)
        if len(q) != 768:
            if len(q) < 768:
                q = np.pad(q, (0, 768 - len(q)), mode='constant')
            else:
                q = q[:768]
        
        q = normalize_vec(q)
        
        scores, ids = search_index(_index, q, k, embeddings=_embeddings)
        results = []
        for score, idx in zip(scores, ids):
            meta_row = None
            try:
                meta_row = _meta.iloc[int(idx)].to_dict()
            except Exception:
                meta_row = {"id": str(idx)}
            explain = mk_explanation(meta_row, score)
            results.append({
                "id": str(meta_row.get('id', idx)),
                "score": float(score),
                "product_display_name": meta_row.get('product_display_name'),
                "image_url": meta_row.get('image_url'),
                "masterCategory": meta_row.get('masterCategory'),
                "baseColour": meta_row.get('baseColour'),
                "short_explain": explain,
            })
        return {"query_id": str(hash(content) & 0xFFFFFFFF), "results": results}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.exception(f"Error in search_image: {e}\n{error_trace}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "search_failure", 
                "details": str(e),
                "type": type(e).__name__
            }
        )


@router.post("/text")
async def search_text(req: TextSearchRequest):
    try:
        if _text_model is None:
            raise HTTPException(status_code=500, detail={"error": "text_model_not_loaded"})
        if _embeddings is None:
            raise HTTPException(status_code=500, detail={"error": "embeddings_not_loaded"})
        if _meta is None or len(_meta) == 0:
            raise HTTPException(status_code=500, detail={"error": "meta_not_loaded"})
            
        q = _text_model.embed_text(req.text)
        
        # Ensure embedding dimension matches stored embeddings (768)
        if len(q) != 768:
            if len(q) < 768:
                q = np.pad(q, (0, 768 - len(q)), mode='constant')
            else:
                q = q[:768]
        
        q = normalize_vec(q)
        
        scores, ids = search_index(_index, q, req.k, embeddings=_embeddings)
        results = []
        for score, idx in zip(scores, ids):
            meta_row = None
            try:
                meta_row = _meta.iloc[int(idx)].to_dict()
            except Exception:
                meta_row = {"id": str(idx)}
            explain = mk_explanation(meta_row, score)
            results.append({
                "id": str(meta_row.get('id', idx)),
                "score": float(score),
                "product_display_name": meta_row.get('product_display_name'),
                "image_url": meta_row.get('image_url'),
                "masterCategory": meta_row.get('masterCategory'),
                "baseColour": meta_row.get('baseColour'),
                "short_explain": explain,
            })
        return {"query_id": str(hash(req.text) & 0xFFFFFFFF), "results": results}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.exception(f"Error in search_text: {e}\n{error_trace}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "search_failure", 
                "details": str(e),
                "type": type(e).__name__
            }
        )
