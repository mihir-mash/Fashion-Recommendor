from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
from PIL import Image
import io
import numpy as np
import logging

from models import loader
from embeddings import load_embeddings, load_meta, get_embedding_by_id
from indexer import load_index, search_index
from utils import image_to_pil, normalize_vec, mk_explanation

router = APIRouter()
logger = logging.getLogger("backend.routes.matching")

_outfit_model = loader.load_outfit_model()
_index = load_index()
_embeddings = load_embeddings()
_meta = load_meta()


class MatchRequest(BaseModel):
    item_id: Optional[str]
    target_type: Optional[str] = None
    k: Optional[int] = 6


@router.post("/outfit")
async def match_outfit(image: Optional[UploadFile] = File(None), item_id: Optional[str] = Form(None), target_type: Optional[str] = Form(None), k: int = Form(6)):
    try:
        if image is not None:
            content = await image.read()
            pil = image_to_pil(content)
            emb = loader.load_image_model().embed_image(pil)
        elif item_id is not None:
            emb = get_embedding_by_id(item_id)
        else:
            raise HTTPException(status_code=400, detail={"error": "missing_input", "details": "Provide image or item_id"})

        emb = normalize_vec(np.array(emb))

        # try outfit model
        try:
            matches = _outfit_model.predict_matches(emb, target_type, k)
        except Exception:
            # fallback: brute-force find items in complementary categories
            scores, ids = search_index(_index, emb, k, embeddings=_embeddings)
            matches = list(zip([str(i) for i in ids], scores))

        results = []
        for mid, score in matches:
            try:
                row = _meta[_meta['id'].astype(str) == str(mid)].iloc[0].to_dict()
            except Exception:
                row = {'id': mid}
            explain = mk_explanation(row, float(score))
            results.append({
                'id': str(row.get('id', mid)),
                'score': float(score),
                'product_display_name': row.get('product_display_name'),
                'image_url': row.get('image_url'),
                'short_explain': explain,
            })
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail={"error": "match_failure", "details": str(e)})
