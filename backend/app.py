from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config import Settings
import routes.search as search
import routes.weather_rec as weather_rec
import routes.matching as matching
import routes.meta as meta
from fastapi.staticfiles import StaticFiles
import os

settings = Settings()

app = FastAPI(title="Fashion Recommender Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development (restrict in production)
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Fashion Recommender Backend")
    # Pre-compute weather recommendations for all supported cities
    try:
        weather_rec._precompute_all_cities(k=6)
    except Exception as e:
        logger.warning(f"Failed to pre-compute weather recommendations: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/models")
def debug_models():
    """Debug endpoint to check model loading status"""
    from models import loader
    from embeddings import load_embeddings, load_meta
    from indexer import load_index
    
    try:
        img_model = loader.load_image_model()
        txt_model = loader.load_text_model()
        outfit_model = loader.load_outfit_model()
        embeddings = load_embeddings()
        meta = load_meta()
        index = load_index()
        
        return {
            "image_model": {
                "loaded": img_model is not None,
                "has_embed_image": hasattr(img_model, 'embed_image') if img_model else False,
                "type": str(type(img_model)) if img_model else None
            },
            "text_model": {
                "loaded": txt_model is not None,
                "has_embed_text": hasattr(txt_model, 'embed_text') if txt_model else False,
                "type": str(type(txt_model)) if txt_model else None
            },
            "outfit_model": {
                "loaded": outfit_model is not None,
                "has_predict_matches": hasattr(outfit_model, 'predict_matches') if outfit_model else False,
                "type": str(type(outfit_model)) if outfit_model else None
            },
            "embeddings": {
                "loaded": embeddings is not None,
                "shape": list(embeddings.shape) if embeddings is not None else None
            },
            "meta": {
                "loaded": meta is not None,
                "rows": len(meta) if meta is not None else 0
            },
            "index": {
                "loaded": index is not None,
                "type": str(type(index)) if index else None
            }
        }
    except Exception as e:
        logger.exception(e)
        return {"error": str(e), "type": type(e).__name__}


# include routers
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(weather_rec.router, prefix="/recommend", tags=["recommend"])
app.include_router(matching.router, prefix="/match", tags=["match"])
app.include_router(meta.router, prefix="", tags=["meta"])

# Serve product images (if present) at /static/images/<id>.(jpg|png)
images_dir = os.path.join(os.path.dirname(__file__), "data", "archive", "images")
if os.path.isdir(images_dir):
    app.mount("/static/images", StaticFiles(directory=images_dir), name="images")
else:
    logger.info(f"Images directory not found: {images_dir} (static mount skipped)")
