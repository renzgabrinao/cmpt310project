from datasets import load_dataset
from formulas import haversine, mag2e
from data import cities
from coast import signed_dist
import geopandas as gpd
from shapely.geometry import Point

earthquake_data = load_dataset("LoneWolfgang/japan-major-earthquakes")
earthquake = earthquake_data["train"].to_pandas()[["lat", "lon", "depthKm", "magnitude"]]

# drop duplicates
earthquake = earthquake.drop_duplicates(
    subset=["lat", "lon", "depthKm", "magnitude"],
    ignore_index=True
)

def cities_within_radius(lat, lon, radius_km=100):
    distances = haversine(lat, lon, cities["lat"].to_numpy(), cities["lng"].to_numpy())
    nearby_cities = cities.loc[distances <= radius_km, "city_clean"].tolist()
    total_population = int(cities.loc[distances <= radius_km, "population"].sum())
    return nearby_cities, total_population



metrics = earthquake.apply(
    lambda row: cities_within_radius(row["lat"], row["lon"]),
    axis=1,
)

def get_coord(lat, lon):

    point_series = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")

    #Project to Web Mercator (EPSG:3857)
    point_proj = point_series.to_crs(epsg=3857)

    # Extract x and y coordinates as floats
    x_coord = point_proj.geometry.x.iloc[0]
    y_coord = point_proj.geometry.y.iloc[0]

    return x_coord, y_coord


# THE ABOVE FUNCTION IS ONLY FOR single points, for predict_cluster if we decide to keep it

# Create a single GeoSeries for ALL points at once
gdf = gpd.GeoSeries(
    [Point(xy) for xy in zip(earthquake["lon"], earthquake["lat"])],
    crs="EPSG:4326",
)
gdf_proj = gdf.to_crs(epsg=6684)
earthquake["x_coord"] = gdf_proj.geometry.x
earthquake["y_coord"] = gdf_proj.geometry.y

earthquake["seismic_energy"] = mag2e(earthquake["magnitude"])
earthquake["nearby_cities"] = [m[0] for m in metrics]
earthquake["nearby_city_count"] = earthquake["nearby_cities"].apply(len)
earthquake["total_nearby_population"] = [m[1] for m in metrics]

earthquake["coast_distance"] = earthquake.apply(
    lambda row: signed_dist(row["lon"], row["lat"]),
    axis=1
)
