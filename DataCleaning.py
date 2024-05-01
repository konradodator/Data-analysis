import pandas as pd

#Wechselkurs von Dollar zu Euro. Wechselkurs vom 22.04.2024
exchange_rate = 0.94

# DataFrame laden
df = pd.read_excel("C:\\Users\Konrad\Desktop\HTW\M1-Computer_Vision\DataCleaning\\uncleaned2_bike_sales.xlsx")

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


# Überprüfen der bereinigten Daten
print("\nBereinigte Daten:")
print(df_cleaned.head())

# Schreiben des bereinigten DataFrames zurück in die Excel-Datei, um die ursprüngliche Datei zu überschreiben
df_cleaned.to_excel("C:\\Users\Konrad\Desktop\HTW\M1-Computer_Vision\DataCleaning\\uncleaned2_bike_sales.xlsx", index=False)
