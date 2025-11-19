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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
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
