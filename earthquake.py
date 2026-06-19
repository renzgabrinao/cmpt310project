from datasets import load_dataset
import numpy as np
from data import cities

earthquake_data = load_dataset("LoneWolfgang/japan-major-earthquakes")

earthquake = earthquake_data["train"].to_pandas()[["lat", "lon", "depthKm", "magnitude"]]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)

    a = np.sin(dLat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d


def cities_within_radius(lat, lon, radius_km=50):
    distances = haversine(lat, lon, cities["lat"].to_numpy(), cities["lng"].to_numpy())
    nearby_cities = cities.loc[distances <= radius_km, "city_clean"].tolist()
    return nearby_cities


earthquake["nearby_cities"] = earthquake.apply(
    lambda row: cities_within_radius(row["lat"], row["lon"]),
    axis=1,
)

earthquake["nearby_city_count"] = earthquake["nearby_cities"].apply(len)
