import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats

# Read the CSV file with embeddings
df = pd.read_csv("Abstract_EMBEDDED.csv")

# Add index column
df['index_number'] = df.index

# Move the index column to the last position
df = df[[col for col in df.columns if col != 'index_number'] + ['index_number']]

# Sample a subset of the data
sample_size = 2148  # Adjust the sample size as needed
df = df.sample(n=sample_size, random_state=42)

# Remove NaN or infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Remove outliers using z-score
z_scores = stats.zscore(df.iloc[:, :-1])  # Exclude the last column from outlier detection
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 4).all(axis=1)  # Adjust the threshold as needed
df = df[filtered_entries]

# Perform K-means clustering with fine-tuned parameters
clustering = KMeans(n_clusters=9, init='k-means++', n_init=40, tol=1e-6, random_state=42).fit(df.iloc[:, :-1])  # Exclude the last column from clustering

# Add the cluster labels to the DataFrame
df["Cluster"] = clustering.labels_

# Save the preprocessed data to a new CSV file
df.to_csv("Abstract_PREPROCESSED.csv", index=False)
