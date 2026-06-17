import json
import sys

import requests

# WMO weather interpretation codes (Open-Meteo).
WEATHER_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather_forecast(location: str) -> dict:
    """Look up today's weather forecast for a given location.

    Uses the Open-Meteo public API (no API key required). The result is a flat
    dict designed to be easy for a downstream agent to consume.

    Args:
        location: Free-form place name, e.g. "Seattle, WA" or "Tokyo".

    Returns:
        On success, a dict with:
          - location: resolved place name (e.g. "Seattle, United States")
          - current_temp_f, temp_max_f, temp_min_f: temperatures (Fahrenheit)
          - precipitation_mm: total precipitation today (mm)
          - precipitation_probability_pct: max precipitation probability today
          - wind_speed_kmh: current wind speed (km/h)
          - humidity_pct: current relative humidity
          - conditions: short human-readable conditions string
        On failure, a dict with a single "error" key describing the problem.
    """
    parts = [p.strip() for p in location.split(",") if p.strip()]
    name = parts[0] if parts else location.strip()
    hints = [p.lower() for p in parts[1:]]

    try:
        geo = requests.get(
            GEOCODE_URL,
            params={"name": name, "count": 5, "language": "en", "format": "json"},
            timeout=10,
        )
        geo.raise_for_status()
        geo_data = geo.json()
        results = geo_data.get("results") or []
        if not results:
            return {"error": f"Could not resolve location: {location!r}"}

        place = results[0]
        if hints:
            for candidate in results:
                fields = " ".join(
                    str(candidate.get(k, ""))
                    for k in ("country", "country_code", "admin1", "admin2")
                ).lower()
                if any(h in fields for h in hints):
                    place = candidate
                    break
        lat, lon = place["latitude"], place["longitude"]

        fc = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,precipitation_probability_max,weather_code"
                ),
                "timezone": "auto",
                "forecast_days": 1,
                "temperature_unit": "fahrenheit",
            },
            timeout=10,
        )
        fc.raise_for_status()
        fc_data = fc.json()

        current = fc_data["current"]
        daily = fc_data["daily"]
        code = current["weather_code"]

        country = place.get("country") or ""
        resolved = f"{place['name']}, {country}".strip().rstrip(",")

        return {
            "location": resolved,
            "current_temp_f": current["temperature_2m"],
            "temp_max_f": daily["temperature_2m_max"][0],
            "temp_min_f": daily["temperature_2m_min"][0],
            "precipitation_mm": daily["precipitation_sum"][0],
            "precipitation_probability_pct": daily["precipitation_probability_max"][0],
            "wind_speed_kmh": current["wind_speed_10m"],
            "humidity_pct": current["relative_humidity_2m"],
            "conditions": WEATHER_CODES.get(code, f"unknown (code {code})"),
        }
    except requests.RequestException as e:
        return {"error": f"Weather API request failed: {e}"}


def main() -> int:
    if len(sys.argv) > 1:
        location = " ".join(sys.argv[1:])
    else:
        location = input("Location: ").strip()

    if not location:
        print("No location provided.", file=sys.stderr)
        return 2

    result = get_weather_forecast(location)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
