#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np

def kmeans_clustering(data, n_clusters=2):
    """Apply K-means clustering and identify anomalies based on distance from centroids."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans

def identify_anomalies(data, kmeans_model, threshold=1.5):
    """Identify anomalies based on distance from cluster centroids."""
    distances = np.min(kmeans_model.transform(data), axis=1)
    return distances > threshold

