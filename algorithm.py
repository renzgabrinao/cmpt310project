import matplotlib.pyplot as plt
from pathlib import Path
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from coast import signed_dist
from earthquake import earthquake, cities_within_radius
from formulas import mag2e

features = earthquake[
    [
        'lat',
        'lon',
        'depthKm',
        'seismic_energy',
        'coast_distance',
        'total_nearby_population',
    ]
]

# train/val split here before scaling
# However a quick google search seems to say that Unsupervised learning does not usually need a train/val split

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# lowk googled this im still not sure how this thing works
# preetty sure it calculates the stddev and avg then scales everything around avg(avg≈0) with the stddev≈1
# the formula is (x - avg)/stddev

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(scaled_features)

earthquake["cluster"] = kmeans.labels_


def predict_cluster(latitude, longitude, depth, magnitude):
    # Derived Features
    _, nearby_population = cities_within_radius(latitude, longitude)
    seismic_energy = mag2e(magnitude)
    coast_distance = signed_dist(longitude, latitude)
    # Bev's mostly unchanged function
    input_features = [[latitude, longitude, depth, seismic_energy, coast_distance, nearby_population]]
    scaled_input = scaler.transform(input_features)
    cluster_label = kmeans.predict(scaled_input)
    return cluster_label[0]

# also googled this, probably needs so much more tweaking


def export_clustered_earthquakes(dataframe, output_path):
    export_df = dataframe.copy()
    export_df["nearby_cities"] = export_df["nearby_cities"].apply(lambda cities: "; ".join(cities))
    export_df.to_csv(output_path, index=False)

    # basically just cleans up the nearby cities list so it looks nicer in the csv, also exports the data to a csv file in output dir

if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    csv_output = output_dir / "earthquakes_clustered.csv"

    export_clustered_earthquakes(earthquake, csv_output)

    print(predict_cluster(35.0, 135.0, 10.0, 6.2))
    # test line
    print(f"Saved clustered data to {csv_output}")