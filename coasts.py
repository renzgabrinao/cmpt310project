
import ssl

import geopandas as gpd
url = "https://geo.vliz.be/geoserver/MarineRegions/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=MarineRegions:coasts_per_ocean&outputFormat=json"

ssl_context = ssl._create_unverified_context()

# Pass the unverified context to geopandas via the storage_options parameter
gdf_coast = gpd.read_file(url, storage_options={"context": ssl_context})

# Save it to your local folder so you never have to download it again
gdf_coast.to_file("coasts_per_ocean.json", driver="GeoJSON")