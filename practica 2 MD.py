import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

#print(df.head())

#print(df.isnull().sum())
#print(df.duplicated().sum())

numeric_cols = ['Score', 'Episodes', 'Rank', 'Popularity', 'Favorites', 'Scored By', 'Members']
print("\nEstadísticas descriptivas para columnas numéricas:")
print(df[numeric_cols].describe())

# Estadísticas para columnas categóricas
categorical_cols = ['Type', 'Status', 'Source', 'Rating', 'Genres', 'Producers', 'Licensors', 'Studios']
print("\nEstadísticas para columnas categóricas:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

