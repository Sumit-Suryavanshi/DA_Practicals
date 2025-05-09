# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output
df = pd.read_csv('D:\\DA practical\\GlobalLandTemperatures_GlobalLandTemperaturesByMajorCity.csv')
# Drop missing values
df = df.dropna()
# Convert 'dt' to datetime format
df['dt'] = pd.to_datetime(df['dt'])
# Convert latitude and longitude to numeric
df['Latitude'] = df['Latitude'].apply(lambda x: float(x[:-1]) * (-1 if x[-1] == 'S' else 1))
df['Longitude'] = df['Longitude'].apply(lambda x: float(x[:-1]) * (-1 if x[-1] == 'W' else 1))
# Extract Latest Records per City
latest_df = df.sort_values('dt').groupby('City').tail(1).copy()
climate_df = latest_df[['City', 'AverageTemperature', 'Latitude', 'Longitude']].copy()
# Normalize Features and Apply K-Means
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(climate_df[['AverageTemperature', 'Latitude', 'Longitude']])
kmeans = KMeans(n_clusters=3, random_state=42)
climate_df['Cluster'] = kmeans.fit_predict(scaled_data)
# Visualize Clusters with Centroids
plt.figure(figsize=(10, 6))
for cluster in range(3):
    cluster_data = climate_df[climate_df['Cluster'] == cluster]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'Cluster {cluster}')
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 2], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')
plt.title('City Clusters Based on Climate')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
# Assign Human-readable Climate Labels
cluster_temps = climate_df.groupby('Cluster')['AverageTemperature'].mean()
cluster_to_climate = {
    cluster_temps.idxmin(): 'Cold',
    cluster_temps.idxmax(): 'Warm',
    cluster_temps.index.difference([cluster_temps.idxmin(), cluster_temps.idxmax()])[0]: 'Moderate'}
climate_df['Climate'] = climate_df['Cluster'].map(cluster_to_climate)
# Visualize Climate Zones
plt.figure(figsize=(10, 6))
colors = {'Cold': 'blue', 'Moderate': 'green', 'Warm': 'red'}
for climate in climate_df['Climate'].unique():
    data = climate_df[climate_df['Climate'] == climate]
    plt.scatter(data['Longitude'], data['Latitude'],
                color=colors[climate], label=climate, alpha=0.6)
plt.scatter(centroids[:, 2], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')
plt.title('City Clusters Based on Climate Zones')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
# Global Heatmap of Average Temperature
climate_df['Lat_round'] = climate_df['Latitude'].round(1)
climate_df['Lon_round'] = climate_df['Longitude'].round(1)
heatmap_data = climate_df.pivot_table(
    index='Lat_round',
    columns='Lon_round',
    values='AverageTemperature',
    aggfunc='mean')
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', cbar_kws={'label': 'Avg Temp (°C)'})
plt.title('Global City Temperature Heatmap')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# Cluster-wise Heatmaps
def plot_cluster_heatmap(climate_type):
    subset = climate_df[climate_df['Climate'] == climate_type]
    pivot_data = subset.pivot_table(
        index='Lat_round',
        columns='Lon_round',
        values='AverageTemperature',
        aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, cmap='coolwarm', cbar_kws={'label': 'Avg Temp (°C)'})
    plt.title(f'{climate_type} Cities Temperature Heatmap')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
for climate in ['Cold', 'Moderate', 'Warm']:
    plot_cluster_heatmap(climate)
# Climate Trend Over Time (Optional Bonus)

df['Year'] = df['dt'].dt.year
city_cluster_map = climate_df[['City', 'Cluster', 'Climate']]
merged_df = df.merge(city_cluster_map, on='City', how='inner')
trend_df = merged_df.groupby(['Year', 'Climate'])['AverageTemperature'].mean().reset_index()
plt.figure(figsize=(14, 6))
sns.lineplot(data=trend_df, x='Year', y='AverageTemperature', hue='Climate')
plt.title('Climate Trend Over Time by Cluster')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.grid(True)
plt.show()
