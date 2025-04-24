import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

#print(df.head())


df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df = df.dropna(subset=['Score', 'Genres'])

# Preprocesamiento de géneros
df['Genres_list'] = df['Genres'].str.split(', ')

# Codificación one-hot de géneros
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df['Genres_list']), 
                            columns=mlb.classes_,
                            index=df.index)

# Combinar con el Score
data = pd.concat([df['Score'], genres_encoded], axis=1)

# Verificar géneros más comunes
print("Géneros más comunes:")
print(genres_encoded.sum().sort_values(ascending=False).head(10))

X = genres_encoded
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluar
r2 = r2_score(y_test, y_pred)
print(f"\nR² Score del modelo: {r2:.3f}")

# Coeficientes
coef_df = pd.DataFrame({'Genre': X.columns, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print("\nTop 10 géneros con mayor impacto positivo:")
print(coef_df.head(10))

plt.figure(figsize=(12, 8))
top_coef = pd.concat([coef_df.head(5), coef_df.tail(5)])
sns.barplot(x='Coefficient', y='Genre', data=top_coef, palette='coolwarm')
plt.title('Géneros con Mayor Impacto en el Score')
plt.xlabel('Coeficiente del Modelo Lineal')
plt.ylabel('Género')
plt.show()



