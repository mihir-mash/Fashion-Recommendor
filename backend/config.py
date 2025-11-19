import os
from pathlib import Path
from typing import Optional


def _load_env_file(env_path: str = ".env") -> None:
    """Simple .env loader: sets variables into os.environ if not already set."""
    p = Path(env_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # don't overwrite existing env vars
        if key not in os.environ:
            os.environ[key] = val


# load .env in backend folder if present
_load_env_file(Path(__file__).parent / '.env')


class Settings:
    """Lightweight settings loader using environment variables.

    Reads values from environment (and a local `backend/.env` file if present).
    This avoids having a hard dependency on pydantic/BaseSettings so the
    project runs on systems with different pydantic versions.
    """

    def __init__(self) -> None:
        self.WEATHER_API_KEY: Optional[str] = os.getenv('WEATHER_API_KEY')
        self.FAISS_INDEX_PATH: str = os.getenv('FAISS_INDEX_PATH', 'backend/models/faiss.index')
        self.EMBEDDINGS_PATH: str = os.getenv('EMBEDDINGS_PATH', 'backend/models/embeddings.npy')
        self.META_PATH: str = os.getenv('META_PATH', 'backend/styles.csv')
        self.IMAGE_MODEL_PATH: str = os.getenv('IMAGE_MODEL_PATH', 'backend/models/image_model.pkl')
        self.TEXT_MODEL_PATH: str = os.getenv('TEXT_MODEL_PATH', 'backend/models/text_model.pkl')
        self.OUTFIT_MODEL_PATH: str = os.getenv('OUTFIT_MODEL_PATH', 'backend/models/outfit_model.pkl')
        self.FRONTEND_ORIGIN: str = os.getenv('FRONTEND_ORIGIN', 'http://localhost:3000')

    def __repr__(self) -> str:  # helpful for debugging
        return (
            f"Settings(WEATHER_API_KEY={'set' if self.WEATHER_API_KEY else 'unset'}, "
            f"FAISS_INDEX_PATH={self.FAISS_INDEX_PATH}, EMBEDDINGS_PATH={self.EMBEDDINGS_PATH}, "
            f"META_PATH={self.META_PATH})"
        )

