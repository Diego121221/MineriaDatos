import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

#print(df.head())

#Funcion para recomendar animes segun su distancia
def recommend_anime(anime_name, n_recommendations=5):
    try:
       
        idx = df_model[df_model['Name'] == anime_name].index[0]
        anime_features = X.iloc[idx:idx+1]
        
        distances, indices = pipeline.named_steps['model'].kneighbors(
            pipeline.named_steps['preprocessor'].transform(anime_features),
            n_neighbors=n_recommendations+1)
        
       
        indices = indices[0][1:]
        distances = distances[0][1:]
        
        recommendations = df_model.iloc[indices][['anime_id', 'Name']]
        recommendations['similarity'] = 1 - distances 
        
        return recommendations
    except:
        return f"No se encontró el anime '{anime_name}' o hubo un error al procesar."


columnas = ['anime_id', 'Name', 'English name', 'Other name', 'Score', 'Genres', 
            'Synopsis', 'Type', 'Episodes', 'Aired', 'Premiered', 'Status', 
            'Producers', 'Licensors', 'Studios', 'Source', 'Duration', 'Rating', 
            'Rank', 'Popularity', 'Favorites', 'Scored By', 'Members']

features = ['Score', 'Type', 'Genres', 'Episodes', 'Source', 'Rating', 'Popularity']

df = df.replace('UNKNOWN', np.nan)
df.dropna()

df_model = df[features + ['anime_id', 'Name']].copy()

df_model['Genres'] = df_model['Genres'].fillna('Unknown') 

df_model['Genres'] = df_model['Genres'].str.split(', ')  # Convertir géneros en lista

# One-hot encoding para géneros
unique_genres = set()
for genres in df_model['Genres'].dropna():
    if isinstance(genres, list):  # Asegurarse que es una lista
        unique_genres.update(genres)
    
for genre in unique_genres:
    df_model[f'Genre_{genre}'] = df_model['Genres'].apply(
        lambda x: 1 if isinstance(x, list) and genre in x else 0)

df_model.drop('Genres', axis=1, inplace=True)

numeric_features = ['Score', 'Episodes', 'Popularity']
for col in numeric_features:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')  
    df_model[col] = df_model[col].fillna(df_model[col].median()) 


categorical_features = ['Type', 'Source', 'Rating']
for col in categorical_features:
    df_model[col] = df_model[col].fillna(df_model[col].mode()[0]) 


categorical_features += [f'Genre_{g}' for g in unique_genres]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute'))
])


X = df_model.drop(['anime_id', 'Name'], axis=1)

pipeline.fit(X)


name = input("Elige un anime ")
print(recommend_anime(name))  
