import matplotlib.pyplot as plt
from pathlib import Path
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from earthquake import earthquake


features = earthquake[['lat', 'lon', 'depthKm', 'magnitude']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(scaled_features)

earthquake["cluster"] = kmeans.labels_


def predict_cluster(latitude, longitude, depth, magnitude):
    input_features = [[latitude, longitude, depth, magnitude]]
    scaled_input = scaler.transform(input_features)
    cluster_label = kmeans.predict(scaled_input)
    return cluster_label[0]


def export_clustered_earthquakes(dataframe, output_path):
    export_df = dataframe.copy()
    export_df["nearby_cities"] = export_df["nearby_cities"].apply(lambda cities: "; ".join(cities))
    export_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    csv_output = output_dir / "earthquakes_clustered.csv"

    export_clustered_earthquakes(earthquake, csv_output)
    print(predict_cluster(35.0, 135.0, 10.0, 6.2))
    print(f"Saved clustered data to {csv_output}")