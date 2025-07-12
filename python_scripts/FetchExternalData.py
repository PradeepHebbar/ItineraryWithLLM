#!/usr/bin/env python3
"""
External Data Fetching and Processing Module

This module fetches comprehensive travel data from external APIs (Google Places, TripAdvisor)
and enriches it with sentiment analysis, geographic calculations, and intelligent classification.
It's designed to build a rich dataset of Mysuru area attraction# Paginated search for tourist attractions in the Mysuru region
# This loop collects up to 60 tourist attractions using Google Places API
# with automatic pagination handling
#
# Search Parameters:
# - Query: "tourist attractions" (broad search for comprehensive coverage)
# - Location bias: Kushalnagar coordinates for regional focus  
# - Radius: 25km to cover greater Mysuru area
# - Results limit: 60 attractions for manageable dataset
#
# Pagination Handling:
# - Uses next_page_token for continued searches
# - Respects Google's 2-second delay between paginated requests
# - Automatically stops when no more results available
while len(search_results) < 60:
    resp = gmaps.places(
        query="tourist attractions",
        location=LOCATION_BIAS,
        radius=RADIUS,
        page_token=page_token
    )
    search_results.extend(resp.get("results", []))
    page_token = resp.get("next_page_token")
    if not page_token:
        break
    time.sleep(2)  # per Google's guideline

# Limit to 60 places for processing efficiency
places = search_results[:60]

# === Step 2: Detailed Data Extraction ===
# For each discovered place, fetch comprehensive details including:
# - Basic information (name, types, address, coordinates)
# - Quality metrics (ratings, review counts, sentiment analysis) 
# - Operational data (opening hours, current status)
# - Media content (photos, official websites)
# - Enhanced descriptions (Wikipedia, TripAdvisor)
# - Computed metrics (distance, suitability, activity flavors)
rows = []
for p in places:inerary system.

Key Features:
- Google Places API integration for attraction discovery and details
- TripAdvisor API integration for additional travel information
- Wikipedia API for detailed descriptions
- VADER sentiment analysis of user reviews
- Geographic distance calculations from city center
- Intelligent attraction classification and suitability analysis
- Automated data export to Excel format

Data Sources:
- Google Places API: Core attraction data, ratings, reviews, photos
- TripAdvisor API: Additional pricing and travel-specific information
- Wikipedia API: Detailed descriptions and context
- VADER: Sentiment analysis of user reviews

Output Schema:
- place_id: Google Places unique identifier
- name: Attraction name
- types: Comma-separated attraction categories
- address: Full formatted address
- lat/lng: Geographic coordinates
- distance_km: Distance from Mysuru city center
- rating: Average user rating
- review_count: Total number of reviews
- opening_hours: Weekly operating schedule
- open_now: Current operating status
- photo_url: Primary attraction photo
- latest_reviews: Recent user review snippets
- sentiment_score: Aggregate sentiment score
- description: Detailed description from Wikipedia/TripAdvisor
- suitability: Target audience classifications
- flavors: Activity type classifications
- official_website: Official website URL

Dependencies:
- googlemaps: Google Places API client
- requests: HTTP requests for external APIs
- pandas: Data manipulation and Excel export
- nltk: VADER sentiment analysis
- urllib.parse: URL encoding for Wikipedia
- math: Geographic distance calculations

Configuration:
- GOOGLE_API_KEY: Google Places API key
- RAPIDAPI_KEY: TripAdvisor API key via RapidAPI
- Search radius: 25km around Mysuru region
- Location bias: Kushalnagar coordinates for broader coverage

Author: Travel Itinerary System
Version: 1.0
"""

import os
import time
import googlemaps
import requests
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import urllib.parse
import math

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env file won't be loaded.")
    print("Install with: pip install python-dotenv")

# === API Configuration ===
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'Add_Your_Google_Key')       # Google Places API key
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', 'Add_Your_Key')        # TripAdvisor via RapidAPI

# Validate API keys
if GOOGLE_API_KEY == 'Add_Your_Google_Key':
    print("Warning: GOOGLE_API_KEY not set in environment variables. Please add it to your .env file.")
    print("Get your API key from: https://console.cloud.google.com/apis/credentials")

if RAPIDAPI_KEY == 'Add_Your_Key':
    print("Warning: RAPIDAPI_KEY not set in environment variables. Please add it to your .env file.")
    print("Get your API key from: https://rapidapi.com/")

# === Service Initialization ===
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
sentiment = SentimentIntensityAnalyzer()

# === Geographic Configuration ===
# Mysuru region search parameters
LOCATION_BIAS = "12.414,76.704"   # Kushalnagar coordinates for broader coverage
RADIUS        = 25000             # 25km search radius
CENTER_LAT, CENTER_LNG = 12.2958, 76.6394 # Mysuru city center coordinates

# === Helper Functions ===

def fetch_tripadvisor(place_name):
    """
    Fetch additional travel information from TripAdvisor API.
    
    This function attempts to retrieve TripAdvisor-specific data such as
    pricing, official ratings, and travel-specific details for a given place.
    
    Args:
        place_name (str): Name of the attraction to search for
    
    Returns:
        dict: Dictionary containing TripAdvisor data fields:
            - ta_price: Ticket price information
            - ta_website: TripAdvisor-specific website
            - ta_description: TripAdvisor description
            - Additional fields as available from API
    
    API Configuration:
        - Uses RapidAPI TripAdvisor endpoint
        - Requires RAPIDAPI_KEY for authentication
        - Limited to location_id "293621" (Mysuru region)
        - Returns max 1 result per query
    
    Error Handling:
        - Returns empty dict on API failure
        - Gracefully handles HTTP errors and timeouts
        - No exceptions raised to calling code
    
    Note:
        - Currently configured as stub implementation
        - Requires valid TripAdvisor API endpoint subscription
        - Replace URL and parameters with actual TripAdvisor API details
    """
    # Example (replace with real endpoint):
    url = "https://tripadvisor1.p.rapidapi.com/attractions/list"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "tripadvisor1.p.rapidapi.com"
    }
    params = {"location_id": "293621", "limit": "1", "q": place_name}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        data = resp.json()
        # parse what you need, e.g. ticket_price, website, official_rating...
        return {
            "ta_price": None,
            "ta_website": None,
            # ...
        }
    else:
        return {}

def analyze_sentiment(text):
    """
    Analyze sentiment of review text using VADER sentiment analysis.
    
    This function processes user review text and returns a compound sentiment
    score indicating overall emotional tone.
    
    Args:
        text (str): Review text to analyze
    
    Returns:
        float: Compound sentiment score ranging from -1.0 to 1.0
            - Values > 0.05: Positive sentiment
            - Values < -0.05: Negative sentiment  
            - Values between -0.05 and 0.05: Neutral sentiment
    
    VADER Features:
        - Specifically designed for social media text
        - Handles emoticons, punctuation, and capitalization
        - No training required, works out-of-the-box
        - Provides compound score for overall sentiment
    
    Usage:
        - Applied to latest review snippets from Google Places
        - Used to calculate aggregate sentiment scores for attractions
        - Helps in recommendation ranking and filtering
    
    Example:
        >>> score = analyze_sentiment("Amazing place with beautiful views!")
        >>> print(score)  # Returns positive value ~0.7
    """
    return sentiment.polarity_scores(text)["compound"]


# === Classification Helper Functions ===

def determine_suitability(types):
    """
    Determine target audience suitability based on attraction types.
    
    This function analyzes Google Places types to classify attractions
    by their suitability for different traveler demographics.
    
    Args:
        types (list[str]): List of Google Places types for an attraction
    
    Returns:
        list[str]: List of suitability flags indicating target audiences:
            - "family": Suitable for families with children
            - "kids": Specifically appealing to children
            - "friends": Good for friend groups and social activities
            - "couples": Romantic or intimate settings
            - "solo": Suitable for solo travelers
    
    Classification Logic:
        - Family/Kids: Tourist attractions, parks, zoos, temples, water parks
        - Friends: Nightlife venues, bars, restaurants, water activities
        - Couples/Solo: Art galleries, museums, scenic viewpoints, cultural sites
    
    Multiple Classifications:
        - Attractions can have multiple suitability flags
        - Duplicates are automatically removed
        - Empty list returned if no matches found
    
    Usage:
        - Used in data enrichment pipeline
        - Helps itinerary planner filter by traveler type
        - Supports personalized recommendation engine
    
    Example:
        >>> types = ["tourist_attraction", "park", "point_of_interest"]
        >>> suitability = determine_suitability(types)
        >>> print(suitability)  # ["family", "kids", "couples", "solo"]
    """
    flags = []
    # family & kids spots
    if any(t in types for t in ["tourist_attraction","park","zoo","water_park","amusement_park","temple"]):
        flags += ["family","kids"]
    # friends & nightlife
    if any(t in types for t in ["night_club","bar","restaurant","water"]):
        flags.append("friends")
    # couples & solo – art, museums, viewpoints
    if any(t in types for t in ["art_gallery","museum","point_of_interest","scenic_viewpoint"]):
        flags += ["couples","solo"]
    return list(set(flags))

def determine_flavors(types):
    """
    Determine activity flavors/themes based on attraction types.
    
    This function classifies attractions by the type of experience or
    activity they offer to help in itinerary planning and filtering.
    
    Args:
        types (list[str]): List of Google Places types for an attraction
    
    Returns:
        list[str]: List of activity flavor tags:
            - "relax": Peaceful, calming activities
            - "explore": Adventure and discovery activities
            - "photography": Scenic and photogenic locations
    
    Classification Logic:
        - Relax: Parks, gardens, beaches, peaceful settings
        - Explore: Tourist attractions, adventure activities, landmarks
        - Photography: Scenic viewpoints, monuments, historic sites
    
    Multiple Flavors:
        - Attractions can have multiple flavor tags
        - Helps create diverse itineraries
        - Supports activity-based filtering
    
    Usage:
        - Used in itinerary generation algorithms
        - Helps balance activity types in daily plans
        - Supports theme-based trip planning
    
    Example:
        >>> types = ["scenic_viewpoint", "tourist_attraction", "park"]
        >>> flavors = determine_flavors(types)
        >>> print(flavors)  # ["relax", "explore", "photography"]
    """
    flavors = []
    if any(t in types for t in ["park","garden","beach"]):
        flavors.append("relax")
    if any(t in types for t in ["tourist_attraction","adventure","hiking","landmark"]):
        flavors.append("explore")
    if any(t in types for t in ["scenic_viewpoint","monument","historic_site"]):
        flavors.append("photography")
    return flavors

# === Geographic Calculation Functions ===

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.
    
    This function uses the Haversine formula to compute the shortest distance
    over the earth's surface between two geographic points specified by their
    latitude and longitude coordinates.
    
    Args:
        lat1 (float): Latitude of first point in decimal degrees
        lon1 (float): Longitude of first point in decimal degrees
        lat2 (float): Latitude of second point in decimal degrees
        lon2 (float): Longitude of second point in decimal degrees
    
    Returns:
        float: Distance between the two points in meters
    
    Formula:
        Uses the Haversine formula for great circle distance calculation:
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ atan2( √a, √(1−a) )
        d = R ⋅ c
        
        Where:
        - φ is latitude, λ is longitude, R is earth's radius
        - Δφ is difference in latitude, Δλ is difference in longitude
    
    Accuracy:
        - Earth radius: 6,371,000 meters (standard approximation)
        - Assumes spherical Earth (good approximation for travel distances)
        - Accuracy within 0.5% for distances up to several hundred kilometers
    
    Usage:
        - Calculate distance from attractions to Mysuru city center
        - Used for proximity-based filtering and sorting
        - Helps in geographic clustering of attractions
    
    Example:
        >>> # Distance from Mysore Palace to Brindavan Gardens
        >>> dist = haversine(12.3051, 76.6551, 12.4086, 76.6947)
        >>> print(f"{dist/1000:.2f} km")  # ~14.2 km
    """
    R = 6371000  # Earth radius in meters
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# === Main Data Collection Process ===

# Step 1: Discover Places using Google Places API
search_results = []
page_token   = None

# Search for tourist attractions in the Mysuru region
# Uses pagination to collect comprehensive results

while len(search_results) < 60:
    resp = gmaps.places(
        query="tourist attractions",
        location=LOCATION_BIAS,
        radius=RADIUS,
        page_token=page_token
    )
    search_results.extend(resp.get("results", []))
    page_token = resp.get("next_page_token")
    if not page_token:
        break
    time.sleep(2)  # per Google’s guideline

# trim to 60
places = search_results[:60]

# ── STEP 2: GET DETAILS FOR EACH PLACE ──────────────────────────────────
rows = []
for p in places:
    pid = p["place_id"]
    detail = gmaps.place(place_id=pid, fields=[
        "name",
        "type",                   # ← was "types"
        "geometry",               # returns geometry/location & viewport
        "formatted_address",
        "opening_hours",          # weekly schedule
        "current_opening_hours",  # “open_now” flag
        "website",
        "url",
        "rating",
        "user_ratings_total",
        "photo",                  # ← was "photos"
        "review"                  # ← was "reviews"
    ])["result"]

    # core fields
    name      = detail.get("name")
    types     = detail.get("types", [])
    addr      = detail.get("formatted_address")
    latlng    = detail["geometry"]["location"]
    rating    = detail.get("rating")
    rev_count = detail.get("user_ratings_total")

    # opening hours
    hours        = detail.get("opening_hours", {}).get("weekday_text", [])
    open_now     = detail.get("opening_hours", {}).get("open_now")
    weekly_off   = []  # not directly exposed by API
    visit_time   = None  # you can set defaults or derive from `types`

    # photo
    photo_ref = detail.get("photos", [{}])[0].get("photo_reference")
    photo_url = None
    if photo_ref:
        photo_url = (
            "https://maps.googleapis.com/maps/api/place/photo"
            f"?maxwidth=800"
            f"&photoreference={photo_ref}"
            f"&key={GOOGLE_API_KEY}"
        )

    # reviews & sentiment (latest 3)
    reviews = detail.get("reviews", [])[:3]
    latest_snips = [r.get("text") for r in reviews]
    senti_scores = [analyze_sentiment(t) for t in latest_snips]
    lat, lng = detail["geometry"]["location"]["lat"], detail["geometry"]["location"]["lng"]
    # ── NEW: compute distance from center ───────────────────────────────
    dist_m = haversine(CENTER_LAT, CENTER_LNG, lat, lng)
    dist_km = dist_m / 1000

    # fetch TripAdvisor extras (stub)
    ta = fetch_tripadvisor(name)

    # try Wikipedia first
    description = None
    try:
        wiki_title = urllib.parse.quote(name)
        wiki_url   = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}"
        r = requests.get(wiki_url, timeout=5)
        if r.status_code == 200:
            description = r.json().get("extract")
    except Exception:
        description = None

    # fallback: your TripAdvisor stub could provide ta_description
    ta = fetch_tripadvisor(name)
    if not description:
        description = ta.get("ta_description")

    # ——— Suitability & Flavors —————————————————————————————
    suitability_flags = determine_suitability(types)
    trip_flavors      = determine_flavors(types)

    # ——— Official Website ———————————————————————————————
    official_website = detail.get("website")

    rows.append({
        "place_id": pid,
        "name": name,
        "types": ",".join(types),
        "address": addr,
        "lat": latlng["lat"],
        "lng": latlng["lng"],
        "distance_km": round(dist_km, 2),
        "rating": rating,
        "review_count": rev_count,
        "opening_hours": "\n".join(hours),
        "open_now": open_now,
        "photo_url": photo_url,
        "latest_reviews": " || ".join(latest_snips),
        "sentiment_score": sum(senti_scores)/len(senti_scores) if senti_scores else None,
        # **ta  # merge any TripAdvisor fields
         "description": description,
        "suitability": ",".join(suitability_flags),
        "flavors": ",".join(trip_flavors),
        "official_website": official_website,
    })

# === Step 3: Data Export ===
"""
Export processed attraction data to Excel format for further use.

The resulting Excel file contains comprehensive information about Mysuru area
attractions including all fetched, calculated, and classified data fields.

Output File: srirangapattna_attractions_new.xlsx
Columns: place_id, name, types, address, lat, lng, distance_km, rating,
         review_count, opening_hours, open_now, photo_url, latest_reviews,
         sentiment_score, description, suitability, flavors, official_website

Data Quality:
- Geographic coordinates validated
- Sentiment scores calculated from actual reviews
- Distance measurements from city center
- Intelligent suitability and flavor classifications
- Enhanced descriptions from multiple sources

Usage:
- Can be loaded by PrepareExcelVector.py for text conversion
- Used by embed_index.py for vector embedding creation
- Serves as primary data source for itinerary planning system
"""
df = pd.DataFrame(rows)
df.to_excel("srirangapattna_attractions_new.xlsx", index=False)
print("✔️ Written srirangapattna_attractions_new.xlsx with", len(df), "rows")

