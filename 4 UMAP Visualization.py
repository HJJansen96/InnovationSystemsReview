import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.preprocessing import StandardScaler

# Read the preprocessed data from the CSV file
df = pd.read_csv("Abstract_CLUSTERED 2002-2007.csv")

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.iloc[:, :-1])  # Exclude the last column from scaling

# Apply UMAP for dimensionality reduction
umap = UMAP(n_components=2, random_state=10)
embeddings_2d = umap.fit_transform(scaled_data)

# Get unique cluster labels
unique_labels = np.unique(df["Cluster"])

# Assign different colors to each cluster
colors = ['#7F7F7F', '#7030A0', '#0070C0', '#00B0F0', '#00B050', '#92D050', '#FFC000', '#C00000'] 

# Plot the clusters with different colors and labels
for label, color in zip(unique_labels, colors):
    indices = df["Cluster"] == label
    # Define a dictionary to map cluster labels to names
    cluster_names = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 7: "7", 8: "8"} 

    # Get the name of the current cluster label
    cluster_name = cluster_names[label]

    # Get the distances of data points from the cluster's center
    distances = np.linalg.norm(embeddings_2d[indices] - np.mean(embeddings_2d[indices], axis=0), axis=1)
    
    # Normalize the distances to the range [0, 1]
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    # Set the transparency (alpha) based on the normalized distances
    alphas = 1 - normalized_distances
    
    # Scatter plot for the current cluster with transparency
    plt.scatter(
        embeddings_2d[indices, 0],
        embeddings_2d[indices, 1],
        c=color,
        alpha=alphas,
        label=cluster_name
    )

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()

# Save the plot as a picture
plt.savefig("Abstract_CLUSTERED.png")

# Display the plot
plt.show()
