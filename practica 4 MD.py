import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

#print(df.head())

#print(df.isnull().sum())
#print(df.duplicated().sum())

#print("Conteo por Tipo:")
#print(df['Type'].value_counts())
#print("\nConteo por Status:")
#print(df['Status'].value_counts())
#print("\nConteo por Rating:")
#print(df['Rating'].value_counts())
#print("\nConteo por Source:")
#print(df['Source'].value_counts())

selected_types = ['TV', 'Movie', 'OVA', 'ONA']
df_selected = df[df['Type'].isin(selected_types)]

df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
print("\nVerificación de tipo de Score:", df['Score'].dtype)

#df['Type'] = pd.to_numeric(df['Type'], errors='coerce')
print("\nVerificación de tipo de Type:", df['Type'].dtype)

df_clean = df.dropna(subset=['Score', 'Type'])
print("\nValores nulos:\n", df_clean.isnull().sum())


print("\nANOVA para Score por Tipo de Anime:")
model = ols('Score ~ C(Type)', data=df_clean).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

print("\nPrueba Tukey HSD para comparaciones por pares:")
tukey = pairwise_tukeyhsd(endog=df_clean['Score'],
                         groups=df_clean['Type'],
                         alpha=0.05)
print(tukey.summary())

# Visualización de los intervalos de confianza
tukey.plot_simultaneous()
plt.title('Comparación de Scores entre Tipos de Anime')
plt.show()
