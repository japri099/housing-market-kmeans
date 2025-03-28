import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from google.colab import files
from io import StringIO

# Upload dataset
dataset_file = files.upload()
file_name = list(dataset_file.keys())[0]

# Load dataset
df = pd.read_csv(file_name)

# Tampilkan beberapa baris pertama
display(df.head())

# Pilih fitur numerik untuk clustering
features = ['House Price Index', 'Rent Index', 'Affordability Ratio', 
            'Mortgage Rate (%)', 'Inflation Rate (%)', 'GDP Growth (%)',
            'Population Growth (%)', 'Urbanization Rate (%)', 'Construction Index']

# Normalisasi data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# **1. K-Means Clustering**
# Menentukan jumlah klaster optimal dengan metode Elbow
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()

# Jalankan K-Means dengan jumlah klaster optimal (misal K=4 berdasarkan elbow method)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

# **2. DBSCAN Clustering**
dbscan = DBSCAN(eps=1.5, min_samples=5)  # Parameter eps dan min_samples dapat disesuaikan
df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# Visualisasi hasil clustering dengan scatter plot
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['House Price Index'], y=df['Rent Index'], hue=df['KMeans_Cluster'], palette='viridis')
plt.title('K-Means Clustering')

plt.subplot(1, 2, 2)
sns.scatterplot(x=df['House Price Index'], y=df['Rent Index'], hue=df['DBSCAN_Cluster'], palette='coolwarm')
plt.title('DBSCAN Clustering')
plt.show()
