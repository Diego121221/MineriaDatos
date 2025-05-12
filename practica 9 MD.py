from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np


df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

df = df.replace('UNKNOWN', np.nan)
df = df.replace('?', np.nan)
df.dropna()

def preprocess_genres(genres_series):
    # Unir todos los géneros y separarlos por comas
    all_genres = ','.join(genres_series.dropna().astype(str))
    
    # Limpiar y dividir los géneros
    genres_list = []
    for genre_group in all_genres.split(','):
        genre_clean = genre_group.strip().title()  # Formato de título
        if genre_clean and genre_clean != 'Nan':
            genres_list.append(genre_clean)
    
    return genres_list

# Obtener lista de géneros procesados
genres_list = preprocess_genres(df['Genres'])

# Contar frecuencia de géneros
genre_counts = Counter(genres_list)

# Crear word cloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='plasma',  # Otros buenos colormaps: 'viridis', 'magma', 'inferno'
    max_words=100,
    max_font_size=150,
    relative_scaling=0.5
).generate_from_frequencies(genre_counts)

# Mostrar el word cloud
plt.figure(figsize=(16, 9))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Géneros de Anime Más Populares', fontsize=20, pad=20)
plt.tight_layout()
plt.show()

# Mostrar los 20 géneros más comunes en una tabla
print("\nTop 20 Géneros de Anime:")
for genre, count in genre_counts.most_common(20):
    print(f"{genre}: {count} apariciones")