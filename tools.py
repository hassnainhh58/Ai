# tools.py
import httpx
import os

# --------------------------------------------------
# Calculator
# --------------------------------------------------

def calculator(operation: str, a: float, b: float) -> float:
    """Perform a basic mathematical calculation."""
    operation = operation.lower().strip()

    if operation in ["add", "+", "plus"]:
        return a + b
    if operation in ["subtract", "-", "minus"]:
        return a - b
    if operation in ["multiply", "*", "x", "×", "times"]:
        return a * b
    if operation in ["divide", "/", "÷"]:
        if b == 0:
            raise ValueError("Calculator error: Cannot divide by zero.")
        return a / b

    raise ValueError(f"Unknown operation: {operation}")


# --------------------------------------------------
# Get weather data
# --------------------------------------------------

def get_weather(city: str) -> dict:
    """Get current weather information for a city using Open-Meteo."""
    city = city.strip()

    if not city or not any(char.isalpha() for char in city):
        return {
            "success": False,
            "city": city,
            "error": "Please enter a valid city name.",
        }

    try:
        # Find city coordinates
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        geocoding_params = {"name": city, "count": 1, "language": "en", "format": "json"}

        response = httpx.get(geocoding_url, params=geocoding_params, timeout=10)
        response.raise_for_status()

        location_data = response.json()
        results = location_data.get("results")

        if not results:
            return {
                "success": False,
                "city": city,
                "error": f"Could not find the city '{city}'.",
            }

        location = results[0]
        latitude = location["latitude"]
        longitude = location["longitude"]

        # Get weather forecast
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        }

        response = httpx.get(weather_url, params=weather_params, timeout=10)
        response.raise_for_status()

        current = response.json()["current"]

        return {
            "success": True,
            "city": city,
            "weather": {
                "temperature": current["temperature_2m"],
                "humidity": current["relative_humidity_2m"],
                "wind_speed": current["wind_speed_10m"],
            },
        }

    except Exception as error:
        return {
            "success": False,
            "city": city,
            "error": f"Could not retrieve weather for '{city}'. Reason: {error}",
        }


# --------------------------------------------------
# Web search
# --------------------------------------------------

def web_search(question: str) -> str:
    """Search the web for current or up-to-date information using OpenRouter."""
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "tools": [{"type": "openrouter:web_search"}],
        "messages": [{"role": "user", "content": question}],
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]