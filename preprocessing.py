import pandas as pd
import matplotlib as plt
import math

planets = pd.read_csv("habitable.csv")
keep = list([])

# Removes columns with 61% or more of data missing
def remove_missing(e):
    count = len(planets[planets[e].isnull()])
    div = count/len(planets)
    # print("69420 ", div)
    if div > 0.61:
        return 1
    else:
        return 0

missing_values = [x for x in planets.columns if remove_missing(x)]
planets = planets.drop(missing_values, axis=1)

columns = list([])
col_names = list([])
cols = [i for i in planets.columns if i not in planets._get_numeric_data().columns]
for j in cols:
  if(len(planets[j].unique()) > 10):
      planets = planets.drop(j, axis=1)

col_names = planets.columns.values
print(col_names)