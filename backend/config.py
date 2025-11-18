from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    WEATHER_API_KEY: Optional[str] = None
    FAISS_INDEX_PATH: str = "backend/data/faiss.index"
    EMBEDDINGS_PATH: str = "backend/data/embeddings.npy"
    META_PATH: str = "backend/data/meta.csv"
    IMAGE_MODEL_PATH: str = "backend/models/image_model.pth"
    TEXT_MODEL_PATH: str = "backend/models/text_model.pth"
    OUTFIT_MODEL_PATH: str = "backend/models/outfit_model.pth"
    FRONTEND_ORIGIN: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
