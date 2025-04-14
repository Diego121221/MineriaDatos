import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\kaggle\anime-dataset-2023.csv")

#print(df.head())

print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())