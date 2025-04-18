import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

#print(df.head())

#print(df.isnull().sum())
#print(df.duplicated().sum())

plt.rcParams['figure.figsize'] = (12, 6)

# Asumimos que df es nuestro DataFrame con los datos de anime
# Crearemos algunas columnas adicionales para el análisis
df['Genres_split'] = df['Genres'].str.split(', ')
df['Studios_split'] = df['Studios'].str.split(', ')
df['Producers_split'] = df['Producers'].str.split(', ')

# 1. Histogramas para variables numéricas
numeric_cols = ['Score', 'Episodes', 'Popularity',]
print("Generando histogramas para variables numéricas...")

for col in numeric_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

# 2. Boxplots por categorías
print("\nGenerando boxplots por categorías...")
categorical_for_boxplot = ['Type', 'Status', 'Source',]

for cat_col in categorical_for_boxplot:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cat_col, y='Score', data=df)
    plt.title(f'Distribución de Score por {cat_col}')
    plt.xticks(rotation=45)
    plt.show()

# 3. Gráficos de dispersión para relaciones entre variables numéricas
print("\nGenerando gráficos de dispersión...")
numeric_pairs = [('Members', 'Score'), ('Favorites', 'Score'), 
                ('Scored By', 'Score'),]

for x, y in numeric_pairs:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=x, y=y, data=df, alpha=0.6)
    plt.title(f'Relación entre {x} y {y}')
    plt.show()

# 4. Diagramas de pastel para variables categóricas
print("\nGenerando diagramas de pastel...")
categorical_for_pie = ['Type', 'Status', 'Source', 'Rating']

for cat_col in categorical_for_pie:
    plt.figure(figsize=(8, 8))
    value_counts = df[cat_col].value_counts()
    value_counts.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribución de {cat_col}')
    plt.ylabel('')
    plt.show()

# 5. Gráficos de barras para los géneros más comunes
print("\nGenerando gráficos de barras para géneros...")
genres_expanded = df.explode('Genres_split')
top_genres = genres_expanded['Genres_split'].value_counts().head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title('Top 15 Géneros de Anime')
plt.xlabel('Cantidad de Animes')
plt.ylabel('Género')
plt.show()