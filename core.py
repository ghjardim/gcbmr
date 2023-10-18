import pandas as pd
import numpy as np

# Importing data
data = pd.read_csv("./dataset/imdb_top_1000.csv", na_values = "?")

# Removing unimportant columns
del data['Poster_Link']
del data['Meta_score']
del data['Certificate']
del data['Gross']

