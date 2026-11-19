import pandas as pd 
df = pd.read_csv('smart_city_traffic_mobility.csv')
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
print(df.head())

