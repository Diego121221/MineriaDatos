import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

numerical_features = ['Score', 'Episodes', 'Rank', 'Popularity', 'Favorites', 'Scored By', 'Members']
categorical_features = ['Type', 'Rating']

df = df.replace('UNKNOWN', np.nan)
df.dropna()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=3, random_state=42))
])

X_preprocessed = preprocessor.fit_transform(df)

optimal_clusters = 4  

final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=optimal_clusters, random_state=42))
])

final_model.fit(df)

df['Cluster'] = final_model.predict(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocessed.toarray() if hasattr(X_preprocessed, 'toarray') else X_preprocessed)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Anime Clusters Visualized in 2D')
plt.xlabel('Dimensión Principal 1')
plt.ylabel('Dimensión Principal 2')
plt.colorbar(label='Cluster')
plt.show()