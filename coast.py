import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import nearest_points, polygonize, unary_union

gdf = gpd.read_file("data/coastline/coasts_per_ocean.shp")
# print(gdf.head())
# print(gdf.columns)
# print(gdf.crs)
# gdf.plot()
# plt.show()

# Useful Notes
# EPSG:3857 often called Web Mercator, unit is in meters not degrees, good for calculating dist
# EPSG:4326 actually uses lot lat coordinate system, better for plotting

jpn = gdf.to_crs(epsg=3857)
coast = jpn.union_all()
land = unary_union(list(polygonize(coast)))


def visualize_sd(point):
    nearest_cl = nearest_points(point, coast)[1]

    # Covert to 4326 for plotting
    jpn_ll = jpn.to_crs(epsg=4326)
    point_ll = gpd.GeoSeries([point], crs=jpn.crs).to_crs(epsg=4326).iloc[0]
    nearest_ll = gpd.GeoSeries([nearest_cl], crs=jpn.crs).to_crs(epsg=4326).iloc[0]

    # Draw plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    jpn_ll.plot(ax=ax1)
    ax1.scatter(point_ll.x, point_ll.y, c="red")
    ax1.scatter(nearest_ll.x, nearest_ll.y, c="blue")
    ax1.plot(
        [point_ll.x, nearest_ll.x],
        [point_ll.y, nearest_ll.y],
    )
    ax1.set_title("Japan")

    jpn_ll.plot(ax=ax2)
    ax2.scatter(point_ll.x, point_ll.y, c="red")
    ax2.scatter(nearest_ll.x, nearest_ll.y, c="blue")
    ax2.plot(
        [point_ll.x, nearest_ll.x],
        [point_ll.y, nearest_ll.y],
    )
    ax2.set_title("Zoomed")

    # Zoom in
    MARGIN = 1 # in degrees of lon and lat
    ax2.set_xlim(min(point_ll.x, nearest_ll.x) - MARGIN, max(point_ll.x, nearest_ll.x) + MARGIN)
    ax2.set_ylim(min(point_ll.y, nearest_ll.y) - MARGIN, max(point_ll.y, nearest_ll.y) + MARGIN)

    plt.show()


def signed_dist(lon, lat, plot=False):
    """
    Takes the longitude and latitude and returns the signed distance from Japan's coastline
    Positive if inland and negative if offshore
    :param lon: Longitude
    :param lat: Latitude
    :param plot: whether to invoke the visualizer
    :return: signed distance to nearest coastline
    """
    p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    p = p.to_crs(jpn.crs)

    point = p.iloc[0]
    dist = point.distance(coast)

    if plot:
        visualize_sd(point)

    return dist if land.contains(point) else -dist


if __name__ == "__main__":
    # Testing
    print(signed_dist(139.93388366699222, 36.14829194853206, True))
    # Should be able to just directly
    # from coast import signed_dist
