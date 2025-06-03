from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def find_optimal_clusters(X_scaled, max_clusters=10):
    """Determine optimal number of clusters using elbow method"""
    wcss = []
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, max_clusters+1), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    return wcss

def apply_kmeans(X_scaled, n_clusters=5):
    """Apply K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    return kmeans.fit_predict(X_scaled)

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    X_scaled, df = preprocess_data('../data/Mall_Customers.csv')
    find_optimal_clusters(X_scaled)
    clusters = apply_kmeans(X_scaled)
    df['Cluster'] = clusters
    df.to_csv('../results/customer_segments.csv', index=False)
