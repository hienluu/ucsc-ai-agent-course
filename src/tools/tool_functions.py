import requests


# Example tool functions for demonstration
def get_weather(city: str) -> str:
    """Get weather for a city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        the weather
    """
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C",
        "Tokyo": "Rainy, 20°C"
    }
    return weather_data.get(city, f"Weather data not available for {city}")

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        a (int): First number
        b (int): Second number
    Returns:
        The sum of a and b
    """
    return a + b


def get_current_time() -> str:
    """Get the current time.
    Args:
       empty 
    Returns:
        The current time as a string
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")


def get_location(
    amenity: str | None = None,
    street: str | None = None,
    city: str | None = None,
    county: str | None = None,
    state: str | None = None,
    country: str | None = None,
    postalcode: str | None = None,
) -> list[dict]:
    """Get latitude and longitude for a given location.

    Args:
        amenity (str | None): Amenity or Point of interest.
        street (str | None): Street name.
        city (str | None): City name.
        county (str | None): County name.
        state (str | None): State name.
        country (str | None): Country name.
        postalcode (str | None): Postal code.

    Returns:
        list[dict]: A list of dictionaries with the latitude and longitude of the given location.
                    Returns an empty list if the location cannot be determined.
    """
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "amenity": amenity,
        "street": street,
        "city": city,
        "county": county,
        "state": state,
        "country": country,
        "postalcode": postalcode,
        "format": "json",
    }
    # Filter out None values from parameters
    params = {k: v for k, v in params.items() if v is not None}

    try:
        response = requests.get(base_url, params=params, headers={"User-Agent": "none"})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching location data: {e}")
        return []


def get_stock_price_from_api(content):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={content['ticker']}&apikey={API_KEY}"
    api_request = requests.get(url)
    return api_request.text


def get_company_overview_from_api(content):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={content['ticker']}&apikey={API_KEY}"
    api_response = requests.get(url)
    return api_response.text


def get_company_news_from_api(content):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={content['tickers']}&limit=20&sort=RELEVANCE&apikey={API_KEY}"
    api_response = requests.get(url)
    return api_response.text


def get_news_with_sentiment_from_api(content):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={content['news_topic']}&limit=20&sort=RELEVANCE&apikey={API_KEY}"
    api_request = requests.get(url)
    return api_request.text

def main() -> None:
    """Run a simple test of the get_location helper."""
    location_query = {
        "amenity": "university",
        "street": "Oxford Street",
        "city": "Berkeley",
        "state": "California",
        "postalcode": "94720",
        "country": "United States",
    }

    print("Querying location with:", location_query)
    results = get_location(**location_query)

    if not results:
        print("No results returned.")
        return

    print(f"Received {len(results)} result(s):")
    for idx, item in enumerate(results[:5], start=1):
        lat = item.get("lat")
        lon = item.get("lon")
        display_name = item.get("display_name")
        print(f"{idx}. {display_name}\n   lat={lat}, lon={lon}")


if __name__ == "__main__":
    main()
    

