import requests
import logging
from config import Settings
from typing import Optional, Dict

logger = logging.getLogger("backend.weather")
settings = Settings()

# simple mapping from temperature to season

def temp_to_season(temp_c: float) -> str:
    if temp_c <= 10:
        return "cold"
    if temp_c <= 20:
        return "mild"
    return "warm"


DEFAULT_CATEGORY_MAP = {
    "cold": ["Coats", "Knitwear", "Jackets"],
    "mild": ["Shirts", "Dresses", "Trousers"],
    "warm": ["Tops", "Shorts", "Swimwear"]
}


def fetch_weather(lat: Optional[float] = None, lon: Optional[float] = None, city: Optional[str] = None) -> Dict:
    if not settings.WEATHER_API_KEY:
        raise RuntimeError("WEATHER_API_KEY not configured in environment")
    params = {"appid": settings.WEATHER_API_KEY, "units": "metric"}
    if city:
        params['q'] = city
    else:
        params['lat'] = lat
        params['lon'] = lon
    resp = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=5)
    resp.raise_for_status()
    return resp.json()


def map_weather_to_season(weather_json: Dict) -> Dict:
    main = weather_json.get('main', {})
    temp = main.get('temp')
    conds = weather_json.get('weather', [{}])
    cond = conds[0].get('main', '') if conds else ''
    season = temp_to_season(temp if temp is not None else 20)
    categories = DEFAULT_CATEGORY_MAP.get(season, [])
    return {"temp": temp, "condition": cond, "season": season, "categories": categories}
