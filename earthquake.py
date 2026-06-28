from datasets import load_dataset
from formulas import haversine
from data import cities

earthquake_data = load_dataset("LoneWolfgang/japan-major-earthquakes")

earthquake = earthquake_data["train"].to_pandas()[["lat", "lon", "depthKm", "magnitude"]]





def cities_within_radius(lat, lon, radius_km=100):
    distances = haversine(lat, lon, cities["lat"].to_numpy(), cities["lng"].to_numpy())
    nearby_cities = cities.loc[distances <= radius_km, "city_clean"].tolist()
    total_population = int(cities.loc[distances <= radius_km, "population"].sum())
    return nearby_cities, total_population



metrics = earthquake.apply(
    lambda row: cities_within_radius(row["lat"], row["lon"]),
    axis=1,
)




earthquake["nearby_cities"] = [m[0] for m in metrics]
earthquake["nearby_city_count"] = earthquake["nearby_cities"].apply(len)
earthquake["total_nearby_population"] = [m[1] for m in metrics]
