# Fashion Recommender Backend

This is a production-style FastAPI backend demo for a fashion recommender. It includes stub models so you can run it locally and swap in real teammate models by placing them under `backend/models/`.

## Setup

1. Create a Python virtual environment and activate it.

On macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables (example):

```bash
setx WEATHER_API_KEY "your_api_key"
```

4. Run the app (from the `backend/` directory):

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Examples

Search by image (multipart form):

```bash
curl -X POST "http://localhost:8000/search/image" -F "image=@/path/to/photo.jpg" -F "k=6"
```

Search by text:

```bash
curl -X POST "http://localhost:8000/search/text" -H "Content-Type: application/json" -d '{"text":"red jacket","k":6}'
```

Weather recommendation:

```bash
curl "http://localhost:8000/recommend/weather?lat=51.5&lon=-0.1&k=6"
```

Match outfit (upload image):

```bash
curl -X POST "http://localhost:8000/match/outfit" -F "image=@/path/to/photo.jpg" -F "k=6"
```

Meta:

```bash
curl "http://localhost:8000/meta"
```

## Integrating Real Models

Place model files under `backend/models/` and replace the stub loader implementations in `backend/models/*.py` or ensure the loader finds your model at the configured path.

- Image model must expose `load_image_model(path) -> model` and `model.embed_image(image) -> np.ndarray`
- Text model must expose `load_text_model(path)` and `model.embed_text(text) -> np.ndarray`
- Outfit model must expose `load_outfit_model(path)` and `model.predict_matches(embedding, target_type, k) -> list`

