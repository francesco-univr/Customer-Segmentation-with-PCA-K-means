import kagglehub
# Download latest version
path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")
print("Path to dataset files:", path)

import pandas as pd
csv_path = "data/marketing_campaign.csv"
df = pd.read_csv(csv_path)
df.head()

df = pd.read_csv(csv_path, sep="\t")
df.head()

df.info()

# Remove rows with null values in Income
df = df.dropna(subset=['Income'])

# Verify that null values have been removed
print("Remaining null values in Income:", df['Income'].isnull().sum())

# Check the new dataset size
print("New dataset shape:", df.shape)

# Convert Dt_Customer column to datetime format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# Find the most recent registration date in the dataset
latest_date = df['Dt_Customer'].max()
print("Most recent registration date in the dataset:", latest_date)

# Create the Age column based on the year 2014
df['Age'] = 2014 - df['Year_Birth']

# Remove the original Year_Birth column
df.drop(columns=['Year_Birth'], inplace=True)
print(df[['Age']].describe())

# Remove customers with Age > 100 years
df = df[df['Age'] <= 100]

# Verify that there are no more outliers
print("Customers with Age > 100 after cleaning:", df[df['Age'] > 100].shape[0])

# Count unique values in the Education column
print(df['Education'].value_counts())

# Create a binary column for Education
df['Education_Basic'] = df['Education'].apply(lambda x: 1 if x == 'Basic' else 0)

# Remove the original Education column
df.drop(columns=['Education'], inplace=True)
print(df['Education_Basic'].value_counts())

# Create a binary column for Marital Status
df['Marital_Single'] = df['Marital_Status'].apply(lambda x: 1 if x in ['Single', 'Alone', 'Widow'] else 0)

# Remove the original Marital_Status column
df.drop(columns=['Marital_Status'], inplace=True)
print(df['Marital_Single'].value_counts())

# Create a new column with the total number of children
df['Total_Children'] = df['Kidhome'] + df['Teenhome']

# Remove the original columns
df.drop(columns=['Kidhome', 'Teenhome'], inplace=True)
print(df['Total_Children'].value_counts())

# Create a new column with the total spending
df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
df.drop(columns=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], inplace=True)

# Create a new column with the total purchase
df['Total_Purchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].sum(axis=1)
df.drop(columns=['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'], inplace=True)

# Create a new column for accepted campaigns
df['Accepted_Campaigns'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
df.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], inplace=True)
print(df[['Accepted_Campaigns']].head(20))

# Remove ID, Z_CostContact, Z_Revenue columns
df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Complain'], inplace=True)

print(df.columns)

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the average number of accepted campaigns per customer
accepted_campaigns_mean = df['Accepted_Campaigns'].mean()
print(f"Average accepted campaigns per customer: {accepted_campaigns_mean:.2f}")

# Calculate the total number of accepted campaigns across all customers
total_accepted_campaigns = df['Accepted_Campaigns'].sum()
print(f"📊 Total number of accepted campaigns: {total_accepted_campaigns}")

# Count how many customers responded to the last campaign (Response = 1)
total_response_last_campaign = df['Response'].sum()
print(f"Number of customers who responded to the last campaign: {total_response_last_campaign}")

# Count how many customers not responded to the last campaign (Response = 0)
num_clients_accepted_last_campaign = df['Response'].sum()
num_clients_not_accepted_last_campaign = len(df) - num_clients_accepted_last_campaign
print(f"📊 Number of customers who did NOT accept the last campaign: {num_clients_not_accepted_last_campaign}")

print("---------------------")

# Distribution of Accepted Campaigns by Customers
plt.figure(figsize=(8, 5))
sns.countplot(x=df['Accepted_Campaigns'], palette="coolwarm")
plt.title("Distribution of Accepted Campaigns by Customers")
plt.xlabel("Number of Accepted Campaigns")
plt.ylabel("Number of Customers")
plt.show()


# Calculate the correlation matrix for Income, Total_Spending, Total_Purchases, Accepted_Campaigns, Recency, and Response
correlation_matrix = df[['Income', 'Total_Spending', 'Total_Purchases', 'Accepted_Campaigns', 'Recency', 'Response']].corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix:: Income, Total_Spending, Total_Purchases, Accepted Campaigns, Recency e Response")
plt.show()

from sklearn.preprocessing import StandardScaler

# Define the normalizer
scaler = StandardScaler()

# Copy the original dataset to avoid permanent modifications
df_scaled = df.copy()

# Select only continuous numerical features for standardization (excluding Response)
numerical_features = ['Income', 'Total_Spending', 'Total_Purchases', 'Accepted_Campaigns', 'Recency']

# Apply standardization
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

# Verify the newly standardized values
print(df_scaled[numerical_features].describe())


df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
df["Customer_Age"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days

from sklearn.decomposition import PCA
import numpy as np
# Instantiate PCA with 2 principal components
pca = PCA(n_components=2, random_state=42)

# Apply PCA only on standardized numerical features
df_pca = pca.fit_transform(df_scaled[numerical_features])

# Create a DataFrame with principal components
df_pca_final = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

expl_var_ratio = pca.explained_variance_ratio_
print("Explained variance by PC1 and PC2:", expl_var_ratio)
print("Cumulative explained variance:", np.cumsum(expl_var_ratio))


# Loadings (PCA coefficients)
loadings = pd.DataFrame(pca.components_, columns=numerical_features, index=['PC1', 'PC2'])

print("PCA Loadings (Contribution of each feature to PC1 and PC2):")
print(loadings)


from sklearn.cluster import KMeans

# List to store inertia (WCSS) for different k values
wcss = []

# Test different values of k (from 1 to 10)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_pca_final[['PC1', 'PC2']])
    wcss.append(kmeans.inertia_)

# Elbow method plot to find the optimal number of clusters
plt.figure(figsize=(11,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Numero di Cluster (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method')
plt.grid()
plt.show()


from sklearn.metrics import silhouette_score

# Test k=3 and k=4 after standardization and PCA
for k in [3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca_final[['PC1', 'PC2']])
    silhouette_avg = silhouette_score(df_pca_final[['PC1', 'PC2']], labels)
    print(f"Silhouette Score per k={k}: {silhouette_avg:.4f}")

# Apply K-Means with k=4
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_pca_final['Cluster'] = kmeans.fit_predict(df_pca_final[['PC1', 'PC2']])
centroids = kmeans.cluster_centers_

# Display the cluster centroids
print("Cluster centroids:\n", centroids)

# Plot the clusters with centroids
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_pca_final['PC1'], y=df_pca_final['PC2'], hue=df_pca_final['Cluster'], palette='viridis', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=50, c='red', edgecolors='black', label="Centroidi")
plt.title(f'K-Means Clustering with {3} Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title="Cluster")
plt.show()

silhouette_avg = silhouette_score(df_pca_final, df_pca_final['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Add clusters to the original dataset
df['Cluster'] = df_pca_final['Cluster']

# Analyze the characteristics of each cluster
cluster_analysis = df.groupby('Cluster')[['Income', 'Total_Spending', 'Total_Purchases', 'Accepted_Campaigns', 'Recency', 'Response']].mean()

# Display the table with the average characteristics of each cluster
print("Cluster Analysis:")
print(cluster_analysis)

# Create a heatmap to visualize differences between clusters
plt.figure(figsize=(8, 5))
sns.heatmap(cluster_analysis, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Average Characteristics of Each Cluster")
plt.show()

