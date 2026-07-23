'''
dbscan.py
Description: An additional unsupervised category-picking model for a baseline to compare against the original KMeans Algorithm chosen 

'''

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import silhouette_score, adjusted_rand_score

from coast import signed_dist
from earthquake import earthquake, cities_within_radius
from formulas import mag2e

# Maintaining the same features to do a comparison

features = earthquake[
    [
        'depthKm',
        'seismic_energy',
        'coast_distance',
        'total_nearby_population',
    ]
].copy()


features["seismic_energy"] = np.log1p(features["seismic_energy"])
features["total_nearby_population"] = np.log1p(features["total_nearby_population"])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_weighted = scaled_features.copy()

# Picking a good value for eps (one of the hyperparameters for DBSCAN)
def plot_eps(data, k = 5):
    knn = NearestNeighbors(n_neighbors = k)
    knn_fit = knn.fit(data)
    d, _ = knn_fit.kneighbors(data)
    d = np.sort(d[:, k-1])

    # Picking a optimal eps value 
    for i in [50, 55, 60, 65, 70, 75, 80, 85, 80, 95]:
        index = int(len(d)*i/100)
        print(f'{i} percentile, eps ={d[index]:.4f}')

    # Plotting the results 
    plt.plot(d)
    plt.ylim(0, np.percentile(d, 95))  # zoom into where most points sit
    plt.xlabel('Data points sorted by distance (d)')
    plt.ylabel(f'Distance from {k}-nn')
    plt.title('KNN Distance plot')
    plt.show()

# Setting EPS and MinPts
EPS = 0.6
min_Pts = 40

# Calling DBSCAN 
dbscan = DBSCAN(eps = EPS, min_samples = min_Pts)
dbscan.fit(scaled_features)

# def predict_cluster(latitude, longitude, depth, magnitude):
#     # Derived Features
#     nearby_cities, total_population = cities_within_radius(latitude, longitude)
#     seismic_energy = mag2e(magnitude)
#     coast_distance = signed_dist(longitude, latitude)

#     # Bev's mostly unchanged function
#     x_coord, y_coord = get_coord(latitude, longitude)
#     input_features = [[x_coord, y_coord, depth, seismic_energy, coast_distance, nearby_population]]
#     scaled_input = scaler.transform(input_features)
#     DBSCAN section
#     knn = NearestNeighbors(n_neighbors = 1).fit(scaled_features)
#     _, idx = knn.kneighbors(scaled_input)
#     clsuter_label = dbscan.label_[idx[0][0]]
#     return cluster_label[0]
#     Included just in case we need it 

def export_clustered_earthquakes(dataframe, output_path):
    export_df = dataframe.copy()
    export_df["nearby_cities"] = export_df["nearby_cities"].apply(lambda cities: "; ".join(cities))
    export_df.to_csv(output_path, index=False)

def compare_cluster_noise():
    dbscan_label = dbscan.labels_
    total_length = len(dbscan_label)
    cluster_size = {}
    noise = 0

    for label in dbscan_label:
        if label == -1:
            noise += 1
        else:
            cluster_size[label] = cluster_size.get(label, 0) + 1

    print(f'DBSCAN: {len(cluster_size)} clusters'
    f'and {noise} noise ({noise/total_length:.1%})/ {total_length}')

    cluster_sort = sorted(cluster_size.items(), key = lambda x: x[1], reverse = True)

    for label, size in cluster_sort:
        print(f'Cluster {label}: {size}, points: {size/total_length:.1%}')

def compare_kmeans():
    from algorithm import kmeans, scaled_features as kmeans_scaled_features

    dbscan_label = dbscan.labels_
    kmeans_label = kmeans.labels_

    # Calculating Silhouette Scores
    kmeans_silh = silhouette_score(kmeans_scaled_features, kmeans_label)
    print(f'K-Means Silhouette Score: {kmeans_silh:.3f}')

    # Calculating DBSCAN Silhouette Scores 
    # Need to exclude label for -1 
    dbscan_filter = dbscan_label != -1
    if dbscan_filter.sum() > 0 and len(set(dbscan_label[dbscan_filter])) > 1:
        dbscan_silh = silhouette_score(scaled_features[dbscan_filter], dbscan_label[dbscan_filter])
        print(f'DBSCAN Silhouette Score: {dbscan_silh:.3f}')
    else:
        print('DBSCAN failed to be computed')

    # Using Adjusted Rand index to check the agreeability between the two groups 
    ari = adjusted_rand_score(kmeans_label, dbscan_label)
    print(f'ARI: {ari:.3f}')

    # Adding a visual scatter plot to show the differences 
    fig, (plot1, plot2) = plt.subplots(1, 2, figsize = (14, 6))
    plot1.scatter(earthquake['x_coord'], earthquake['y_coord'], c = kmeans.labels_, cmap = 'tab10', s=3)
    plot1.set_xlabel('Longitude')
    plot1.set_ylabel('Latitude')
    plot1.set_title('K-Means Cluster Method')

    noise = dbscan.labels_ == -1
    plot2.scatter(earthquake.loc[~noise, 'x_coord'], earthquake.loc[~noise, 'y_coord'], c = dbscan.labels_[~noise], cmap = 'tab10', s=3)
    plot2.scatter(earthquake.loc[noise, 'x_coord'], earthquake.loc[noise, 'y_coord'], c = 'lightgrey', cmap = 'tab10', s=3, label = 'Noise')
    plot2.set_xlabel('Longitutde')
    plot2.set_ylabel('Latitude')
    plot2.set_title('DBSCAN Method')
    plot2.legend()
    plt.savefig('method_comparison.png', dpi = 150)
    plt.show()


if __name__ == "__main__":
    # Calling plot_eps when testing to pick an optimal eps value
    # plot_eps(scaled_features, k = 12)

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    csv_output = output_dir / "earthquakes_dbscan.csv"
    compare_cluster_noise()
    compare_kmeans()
    export_clustered_earthquakes(earthquake, csv_output)

    #print(predict_cluster(35.0, 135.0, 10.0, 6.2))
    # test line
    print(f"Saved DBSCAN clustered data to {csv_output}")