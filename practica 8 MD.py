import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from dateutil import parser
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import seaborn as sns


df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

df = df.replace('UNKNOWN', np.nan)
df = df.replace('?', np.nan)
df.dropna()

df = df[~df['Aired'].str.contains('\?', na=False, regex=True)]

def create_date_features(df):
    """Crea características de fecha a partir de la columna Start_Date"""
    df = df.copy()
    df['Year'] = df['Start_Date'].dt.year
    df['Month'] = df['Start_Date'].dt.month
    df['Day'] = df['Start_Date'].dt.day
    return df

# 2. Procesamiento inicial de fechas (como en soluciones anteriores)
df['Start_Date'] = pd.to_datetime(
    df['Aired'].str.split(' to ').str[0],
    errors='coerce'
)
df = df.dropna(subset=['Start_Date'])

# 3. Crear las características de fecha
df = create_date_features(df)

# 4. Verificar que las columnas existen
print("Columnas disponibles:", df.columns.tolist())
assert all(col in df.columns for col in ['Year', 'Month', 'Day']), \
       "Las columnas de fecha no se crearon correctamente"

# 5. Definir el pipeline correctamente
numeric_features = ['Year', 'Month', 'Day']
target = 'Score'  # Ajusta según tu columna objetivo

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 6. Preparar datos
X = df[numeric_features]
y = df[target]

y = y.dropna()

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Para X
    ('regressor', LinearRegression())
])

# Asegurar que X e y tengan el mismo índice
X, y = X.align(y, axis=0, join='inner')

# Entrenamiento
pipeline.fit(X, y)

# 7. Ejecutar el pipeline (ejemplo con train_test_split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
print("ejecutado correctamente")

y_pred = pipeline.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Línea de perfecta predicción
plt.title('Valores Reales vs Predicciones')
plt.xlabel('Puntuación Real')
plt.ylabel('Puntuación Predicha')
plt.grid(True)
plt.show()