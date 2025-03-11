from requirements import *
from inference import *

df = pd.read_csv("segmentation_features.csv")

feature_cols = ["Volume", "Mean Intensity", "Std Intensity"]
X = df[feature_cols]

# Standardize features (already normalized but ensuring proper scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = [] 
silhouette_scores = []

print("Finding optimal cluster count...")
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot Elbow method results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 10), wcss, marker="o", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-cluster sum of squares)")
plt.title("Elbow Method for Optimal K")

# Plot Silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 10), silhouette_scores, marker="s", linestyle="--", color="r")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal K")
plt.show()

optimal_k = int(input("Enter the optimal number of clusters based on the Elbow/Silhouette method: "))

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="tab10", s=100, edgecolor="k")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"Clustering Visualization with {optimal_k} Clusters")
plt.legend(title="Cluster")
plt.show()

df.to_csv("segmentation_clusters.csv", index=False)
print(" Clustering complete! Results saved to segmentation_clusters.csv")