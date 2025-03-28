# Global Housing Market Clustering

Repository ini melakukan clustering pada dataset pasar perumahan global menggunakan **K-Means** dan **DBSCAN**. Data mencakup indikator ekonomi seperti:

- House Price Index
- Rent Index
- Affordability Ratio
- Mortgage Rate (%)
- Inflation Rate (%)
- GDP Growth (%)
- Population Growth (%)
- Urbanization Rate (%)
- Construction Index

## Metode

### 1. K-Means Clustering
Menentukan jumlah klaster optimal menggunakan *Elbow Method* sebelum menjalankan algoritma **K-Means**.

### 2. DBSCAN Clustering
Menggunakan algoritma **Density-Based Spatial Clustering** untuk menemukan pola berdasarkan kepadatan data.

## Instalasi dan Penggunaan

Jalankan skrip ini di **Google Colab**:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from google.colab import files
```

## Hasil Visualisasi
Plot di bawah ini menunjukkan hasil clustering dengan K-Means dan DBSCAN:

![Clustering Visualization](https://github.com/japri099/housing-market-kmeans/blob/main/Clustering%20Visualization.png)

## Download Hasil
Dataset yang telah diklasterisasi dapat diunduh setelah pemrosesan selesai.
