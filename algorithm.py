from pathlib import Path
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from earthquake import earthquake


features = earthquake[
    [
        'x_coord',
        'y_coord',
        'depthKm',
        'seismic_energy',
        'coast_distance',
        'total_nearby_population',
    ]
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

features_df = pd.DataFrame(
    scaled_features, columns=features.columns, index=features.index
)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(features_df)

earthquake["cluster"] = kmeans.labels_


# def predict_cluster(latitude, longitude, depth, magnitude):
#     # Derived Features
#     nearby_cities, total_population = cities_within_radius(latitude, longitude)
#     seismic_energy = mag2e(magnitude)
#     coast_distance = signed_dist(longitude, latitude)

#     # Bev's mostly unchanged function
#     x_coord, y_coord = get_coord(latitude, longitude)
#     input_features = [[x_coord, y_coord, depth, seismic_energy, coast_distance, nearby_population]]
#     scaled_input = scaler.transform(input_features)
#     cluster_label = kmeans.predict(scaled_input)
#     return cluster_label[0]


# IMPORTANT : DONT NEED PREDICT CLUSTER SINCE WE'RE JUST RUNNING KMEANS ON THE DATASET(?)
# also googled this, probably needs so much more tweaking


def export_clustered_earthquakes(dataframe, output_path):
    export_df = dataframe.copy()
    export_df["nearby_cities"] = export_df["nearby_cities"].apply(lambda cities: "; ".join(cities))
    export_df.to_csv(output_path, index=False)

    # basically just cleans up the nearby cities list so it looks nicer in the csv, also exports the data to a csv file in output dir

if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    csv_output = output_dir / "earthquakes_clustered.csv"

    export_clustered_earthquakes(earthquake, csv_output)

    # print(predict_cluster(35.0, 135.0, 10.0, 6.2))
    # test line
    print(f"Saved clustered data to {csv_output}")