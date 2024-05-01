import pandas as pd
import random
import numpy as np
#Wechselkurs von Dollar zu Euro. Wechselkurs vom 22.04.2024
exchange_rate = 0.94

# DataFrame laden
df = pd.read_csv("C:\\Users\Konrad\Desktop\HTW\M1-Computer_Vision\DataCleaning\\uncleaned2_bike_sales.csv")

# Überprüfen der Kopfzeile nach Dollar-Zeichen
dollar_columns = []
for column in df.columns:
    if '$' in column:  # Überprüfen, ob Dollar-Zeichen in der Kopfzeile der Spalte enthalten ist
        dollar_columns.append(column)

print("Hier stehen die Features mnit $ Zeichen:")
print(dollar_columns)

# Umwandeln der Werte in den Spalten mit Dollar-Zeichen in Euro
for column in dollar_columns:
    df[column] = df[column] * exchange_rate

# Anzeigen der ersten paar Zeilen des umgewandelten DataFrames
print("Erste paar Zeilen des umgewandelten DataFrames:")
print(df.head())

# Überblick über die Datentypen und die Anzahl der Nicht-Null-Einträge
print("\nÜberblick über die Datentypen und Nicht-Null-Einträge:")
print(df.info())

# Schritt 2: Data Cleaning
# Auffinden von Spalten mit mehr als 60% fehlenden Einträgen
missing_threshold = len(df) * 0.6
columns_to_drop = df.columns[df.isnull().sum() > missing_threshold]

# Löschen dieser Spalten
df_cleaned = df.drop(columns=columns_to_drop)

# Auffinden von Zeilen mit mehr als 60% fehlenden Einträgen
rows_to_drop = df_cleaned[df_cleaned.isnull().sum(axis=1) > missing_threshold].index

# Löschen dieser Zeilen
df_cleaned = df_cleaned.drop(index=rows_to_drop)

#Schritt 3.
# Stratified replacement für Spalte 2
# Fehlende Werte in Spalte 2 werden durch die nächsthöhere Zahl ersetzt
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 1]):  # Überprüfen, ob der Wert in Spalte 2 fehlt
        df_cleaned.iloc[i, 1] = df_cleaned.iloc[i-1, 1] + 1  # Ersetzen des fehlenden Werts durch die nächsthöhere Zahl

# Datumsersetzung für Spalte 3 und 4 (Stratified replacement)
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 2]):  # Überprüfen, ob das Datum in Spalte 3 fehlt
        df_cleaned.iloc[i, 2] = df_cleaned.iloc[i-1, 2]  # Ersetzen des fehlenden Datums durch das Datum aus der vorherigen Zeile
    if pd.isnull(df_cleaned.iloc[i, 3]):  # Überprüfen, ob der Tag in Spalte 4 fehlt
        # Ersetzen des Tags aus dem Datum der vorherigen Zeile und Zuweisen in Spalte 4
        df_cleaned.iloc[i, 3] = df_cleaned.iloc[i-1, 3]

# Ergenzen des Monats in Spalte 5 (Stratified replacement)
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 4]): # Überprüfen, ob der Monat in Spalte 5 fehlt
        df_cleaned.iloc[i, 4] = df_cleaned.iloc[i-1, 4] #Ersetzen des fehlenden Monats mit dem Monat aus der vorherigen Zeile

# Ergenzen des Jahres in Spalte 6 (Stratified replacement)
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 5]): # Überprüfen, ob des Jahres in Spalte 6 fehlt
        df_cleaned.iloc[i, 5] = df_cleaned.iloc[i-1, 5] #Ersetzen des fehlenden Jahres mit dem Jahr aus der vorherigen Zeile

#Fill with mean für Spalte 7 - Alter der Kunden
#Berechnen des Mittelwerts für Spalte 7
mean_column_7 = df_cleaned.iloc[:, 6].mean()

#Ersetzen fehlender Einträge in Spalte 7 durch den Mittelwert
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 6]):  # Überprüfen, ob der Eintrag in Spalte 7 fehlt
        df_cleaned.iloc[i, 6] = mean_column_7  # Ersetzen des fehlenden Eintrags durch den Mittelwert



# Füllen von Spalte 8 (Customer_Gender) mit random replacement
for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 7]):  # Überprüfen, ob der Eintrag in Spalte 8 fehlt
        # Zufällig 'M' oder 'F' auswählen
        random_gender = random.choice(['M', 'F'])
        df_cleaned.iloc[i, 7] = random_gender


for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 8]): # Überprüfen, ob des Jahres in Spalte 9 fehlt
        df_cleaned.iloc[i, 8] = df_cleaned.iloc[i+1, 8] #Ersetzen der fehlenden Country aus der oberen Zeile

for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 9]): # Überprüfen, ob des Jahres in Spalte 10 fehlt
        df_cleaned.iloc[i, 9] = df_cleaned.iloc[i+1, 9] #Ersetzen des fehlenden State aus der oberen Zeile

for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 10]): # Überprüfen, ob des Jahres in Spalte 9 fehlt
        df_cleaned.iloc[i, 10] = df_cleaned.iloc[i+1, 10] #Ersetzen der fehlenden Country aus der oberen Zeile

for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 11]): # Überprüfen, ob des Jahres in Spalte 10 fehlt
        df_cleaned.iloc[i, 11] = df_cleaned.iloc[i+1, 11] #Ersetzen des fehlenden State aus der oberen Zeile


# Überprüfen der bereinigten Daten
print("\nBereinigte Daten:")
print(df_cleaned.head())

# Schreiben des bereinigten DataFrames zurück in die Excel-Datei, um die ursprüngliche Datei zu überschreiben
df_cleaned.to_excel("C:\\Users\Konrad\Desktop\HTW\M1-Computer_Vision\DataCleaning\\uncleaned2_bike_sales.xlsx", index=False)
