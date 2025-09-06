# task2_kmeans_mall_scaled.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

X = df.iloc[:, [3, 4]].values  # Annual Income vs Spending Score

# Scale the data (important to make clusters compact and neat)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Cluster mapping with fixed colors
cluster_map = {
    0: ("red", "Cluster 1"),    
    1: ("blue", "Cluster 2"),   
    2: ("green", "Cluster 3"),  
    3: ("cyan", "Cluster 4"),   
    4: ("magenta", "Cluster 5") 
}

# Plot clusters
plt.figure(figsize=(10, 7))
for cluster_id, (color, label) in cluster_map.items():
    plt.scatter(
        X[y_kmeans == cluster_id, 0], X[y_kmeans == cluster_id, 1],
        s=50, c=color, label=label, edgecolor='k', alpha=0.8
    )

# Plot centroids (smaller yellow circles)
centers = scaler.inverse_transform(kmeans.cluster_centers_)  # back to original scale
plt.scatter(
    centers[:, 0], centers[:, 1],
    s=100, c='yellow', edgecolor='black', marker='o', label='Centroids'
)

# Formatting
plt.title("K-Means Clustering of Mall Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
