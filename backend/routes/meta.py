from fastapi import APIRouter, HTTPException
import logging
from embeddings import load_meta

router = APIRouter()
logger = logging.getLogger("backend.routes.meta")

_meta = load_meta()


@router.get("/meta")
def meta():
    try:
        df = _meta
        if df is None or df.empty:
            return {"filters": {}}
        filters = {}
        for col in ['masterCategory', 'baseColour', 'fabric']:
            if col in df.columns:
                filters[col] = sorted([str(x) for x in df[col].dropna().unique().tolist()])
        # season tags: optional column
        if 'season' in df.columns:
            filters['season'] = sorted([str(x) for x in df['season'].dropna().unique().tolist()])
        return {"filters": filters}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail={"error": "meta_failure", "details": str(e)})
