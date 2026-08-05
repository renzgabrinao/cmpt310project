import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from algorithm import features_df

def evaluate_kmeans(data, min_k=2, max_k=10, sample_size=10000):
    results = []

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(data)

        score = silhouette_score(
            data,
            labels,
            sample_size=min(sample_size, len(data)),
            random_state=42,
        )

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": score,
        })

        print(
            f"k={k}, "
            f"inertia={model.inertia_:.2f}, "
            f"silhouette={score:.4f}"
        )

    return pd.DataFrame(results)

def plot_results(results):
    plt.figure(figsize=(8, 5))
    plt.plot(results["k"], results["inertia"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.xticks(results["k"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("result/kmeans_elbow.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        results["k"],
        results["silhouette_score"],
        marker="o",
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.xticks(results["k"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("result/kmeans_silhouette.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    results = evaluate_kmeans(features_df)

    print("\nResults:")
    print(results.to_string(index=False))

    best_row = results.loc[
        results["silhouette_score"].idxmax()
    ]

    print(
        f"\nHighest silhouette score: "
        f"k={int(best_row['k'])}, "
        f"score={best_row['silhouette_score']:.4f}"
    )

    results.to_csv(
        "result/kmeans_evaluation.csv",
        index=False,
    )

    plot_results(results)