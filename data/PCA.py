import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

'''
I will be using the Parcel Export Sheet 2 data for the time being as Joseph had stated 
that best case scenario this is the data used to predict with. While he had mentioned 
that some of the data is out of date, I wanted to see what insights could be pulled from 
this first, then work with the other datasets later on as we get a better understanding 
of the data. 
'''

# Read in Parcel Export Sheet 2
parcel_export = pd.read_excel("Parcel Export.xlsx", sheet_name="Full Parcel Summary")
parcel_export.head()
parcel_export.shape # (53277, 53)
parcel_export.info()

# Non-numeric columns
non_numeric_cols = parcel_export.select_dtypes(exclude=[np.number]).columns.tolist()
parcel_export[non_numeric_cols].shape # (53277, 12)
parcel_export[non_numeric_cols].info()
parcel_export[non_numeric_cols].columns

# Numeric columns
numeric_parcel_cols = parcel_export.select_dtypes(include=[np.number]).columns.tolist() 
parcel_export[numeric_parcel_cols].shape # (53277, 41)
parcel_export[numeric_parcel_cols].info() # Identifyed columns with a lot of missing values to drop
parcel_export[numeric_parcel_cols].columns # List of numeric columns for copy/paste

# Filter Parcel Export for PCA (only numeric columns)
numeric_parcel_data = parcel_export[numeric_parcel_cols].drop(
    columns = ['BsmtFinishPct_Weighted','Count_DetGarage', 'SqFt_DetGarage', 'Count_ DetCarport', 
        'SqFt_DetCarport', 'Count_Barn', 'Sqft_Barn', 'Count_Guest_House', 'SqFt_Guest_House', 
        'Count_Pools', 'SqFt_Pools', 'Count_Recreational_Courts', 'SqFt_Recreational_Courts', 
        'Count_Shed', 'SqFt_Shed', 'Count_Gazebo_Pavilion', 'SqFt_Gazebo_Pavilion'])
numeric_parcel_data.shape # (53277, 24)
numeric_parcel_data.info()

# Drop rows with missing values
numeric_parcel_data1 = numeric_parcel_data.dropna()
numeric_parcel_data1.shape # (21440, 24)

# Scaling the data for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_parcel_data1)

# PCA and Scree Plot
pca_full = PCA(n_components=None) # n_components=None keeps all features as components
pca_full.fit(X_scaled)

# Scree Plot to optimize number of components
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         pca_full.explained_variance_ratio_.cumsum(), 
         marker='o', linestyle='--')
plt.title('Scree Plot (Cumulative Variance)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# PCA with 10 components
pca = PCA(n_components=10)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=[
    'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

pca.explained_variance_ratio_.sum()

# TNSE on PCA components
tsne = TSNE(n_components=2, random_state=42, n_iter=5000)
tsne_results = tsne.fit_transform(pca_df)

plt.figure(figsize=(8,6))
plt.scatter(tsne_results[:,0], tsne_results[:,1], alpha=0.5)
plt.title('T-SNE Visualization of Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid()
plt.show()

# Optimize K for K-Means using Elbow Method
wcss = []
for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(pca_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# K-Means Clustering and Visualization for K=2 to K=5
for i in range(2,6):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10) 
    cluster_labels = kmeans.fit_predict(pca_df)

    tsne_df = pd.DataFrame(tsne_results, columns=['tSNE Dim 1', 'tSNE Dim 2'])
    tsne_df['Cluster'] = cluster_labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=tsne_df, 
        x='tSNE Dim 1', 
        y='tSNE Dim 2', 
        hue='Cluster', 
        palette=sns.color_palette('Accent', n_colors=6), 
        legend='full',
        alpha=0.7)
    plt.title('T-SNE Visualization Colored by K-Means Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.show()

# Relate back to original features for K=2 to K=5
for i in range (2,6):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_df)
    numeric_parcel_data1['Cluster'] = cluster_labels

    # Distribution across Clusters
    cluster_counts = numeric_parcel_data1['Cluster'].value_counts()
    print("Cluster Counts:")
    print(cluster_counts)

    # Calculate and compare the mean of all features for each cluster
    cluster_profiles = numeric_parcel_data1.groupby('Cluster').mean().T
    print(cluster_profiles)
