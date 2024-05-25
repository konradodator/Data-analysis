import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#Wechselkurs von Dollar zu Euro. Wechselkurs vom 22.04.2024
exchange_rate = 0.94

# DataFrame laden
df = pd.read_csv('uncleaned2_bike_sales.csv')

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
    if pd.isnull(df_cleaned.iloc[i, 8]): # Überprüfen, ob Country in Spalte 9 fehlt
        df_cleaned.iloc[i, 8] = df_cleaned.iloc[i+1, 8] #Ersetzen der fehlenden Country

for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 9]): # Überprüfen, ob State Spalte 10 fehlt
        df_cleaned.iloc[i, 9] = df_cleaned.iloc[i+1, 9] #Ersetzen des fehlenden State

for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 10]): # Überprüfen, ob Product_Category fehlt
        df_cleaned.iloc[i, 10] = df_cleaned.iloc[i+1, 10] #Ersetzen der fehlenden Product_Category


for i in range(len(df_cleaned)):
    if pd.isnull(df_cleaned.iloc[i, 11]): # Überprüfen, ob Sub_Category fehlt
        df_cleaned.iloc[i, 11] = df_cleaned.iloc[i+1, 11] #Ersetzen des fehlenden Sub_Category




# Erstellen von Boxplots für jedes Feature
for column in df_cleaned.columns:
    if df_cleaned[column].dtype in ['int64', 'float64']:  # Nur numerische Spalten berücksichtigen
        plt.figure(figsize=(8, 6))
        df_cleaned.boxplot(column=[column])
        plt.title(f'Boxplot für {column}')
        plt.ylabel('Wert')
        plt.show()

# Überprüfen der bereinigten Daten
print("\nBereinigte Daten:")
print(df_cleaned.head())


# Schreiben des bereinigten DataFrames zurück in die Excel-Datei, um die ursprüngliche Datei zu überschreiben
df_cleaned.to_csv('bike_sales_clean.csv', index=False)

# Zählen der Anzahl von Männern und Frauen, die ein Fahrrad gekauft haben
gender_counts = df_cleaned['Customer_Gender'].value_counts()

# Erstellen des Balkendiagramms
plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar', color=['blue', 'pink'])
plt.title('Anzahl der Fahrradkäufe nach Geschlecht')
plt.xlabel('Geschlecht')
plt.ylabel('Anzahl der Käufe')
plt.xticks(rotation=0)  # Rotation der x-Achsenbeschriftungen auf 0 Grad
plt.show()

# Gruppieren nach Land und Berechnen des durchschnittlichen Gewinns pro Land
profit_per_country = df_cleaned.groupby('Country')[' Profit_$ '].mean().sort_values()

# Erstellen des Balkendiagramms
plt.figure(figsize=(10, 6))
profit_per_country.plot(kind='bar', color='skyblue')
plt.title('Durchschnittlicher Gewinn pro Land')
plt.xlabel('Land')
plt.ylabel('Durchschnittlicher Gewinn')
plt.xticks(rotation=45, ha='right')  # Rotation der x-Achsenbeschriftungen und Ausrichtung nach rechts
plt.tight_layout()  # Optimierung der Layout-Anpassung
plt.show()

# Erstellen eines Streudiagramms für Männer und Frauen getrennt nach Alter und Umsatz
plt.figure(figsize=(10, 6))

# Streudiagramm für Männer
plt.scatter(df_cleaned[df_cleaned['Customer_Gender'] == 'M']['Customer_Age'],
            df_cleaned[df_cleaned['Customer_Gender'] == 'M']['Revenue_$'],
            color='blue', label='Männer')

# Streudiagramm für Frauen
plt.scatter(df_cleaned[df_cleaned['Customer_Gender'] == 'F']['Customer_Age'],
            df_cleaned[df_cleaned['Customer_Gender'] == 'F']['Revenue_$'],
            color='pink', label='Frauen')

plt.title('Ausgaben nach Alter und Geschlecht')
plt.xlabel('Alter')
plt.ylabel('Umsatz')
plt.legend()
plt.grid(True)
plt.show()


# Anwenden der One-Hot-Codierung auf nicht-numerische Features
df_encoded = pd.get_dummies(df_cleaned, columns=['Month', 'Customer_Gender', 'Country', 'State', 'Product_Category', 'Sub_Category', 'Date'])

# Anzeigen der ersten paar Zeilen des neuen DataFrames
print("Dataframe nach der Codierung:")
print(df_encoded.head())
# Speichern des codierten Datensatzes als CSV-Datei
df_encoded.to_csv('bike_sales_codified.csv', index=False)


# Berechnen der Korrelationsmatrix
correlation_matrix = df_encoded.corr()

# Finden redundanter Features (mit Korrelation > 0.9)
redundant_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            redundant_features.add(correlation_matrix.columns[i])

# Anzeigen der Korrelationsmatrix
print("Korrelationsmatrix:")
print(correlation_matrix)

# Löschen der redundanten Features
df_reduced = df_encoded.drop(columns=redundant_features)

# Löschen der Spalten "Day", "Month" und "Year", da "Date" als information ausreicht
df_cleaned_without_date = df_cleaned.drop(columns=['Day', 'Month', 'Year'])



# Umwandeln aller nicht-numerischen Features in numerische Features
df_cleaned_encoded = pd.get_dummies(df_cleaned)
# Entfernen von Zeilen mit fehlenden Werten
df_cleaned_encoded_no_missing = df_cleaned_encoded.dropna()
# Überprüfen und Entfernen von NaN-Werten
print("Anzahl der NaN-Werte vor der Entfernung:", df_cleaned_encoded.isna().sum().sum())
df_cleaned_encoded_no_missing = df_cleaned_encoded.dropna()
print("Anzahl der NaN-Werte nach der Entfernung:", df_cleaned_encoded_no_missing.isna().sum().sum())

# Standardisieren der Daten
scaler = StandardScaler()
df_standardized_no_missing = scaler.fit_transform(df_cleaned_encoded_no_missing)

# Durchführung der PCA
pca = PCA()
pca.fit(df_standardized_no_missing)

# Berechnen der kumulativen erklärten Varianz
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)

# Bestimmen der Anzahl der Hauptkomponenten, die 95% der Varianz erklären
n_components_95_variance = np.argmax(explained_variance_ratio_cumulative >= 0.95) + 1

# Anzeigen der Anzahl der Principal Components, die 95% der Varianz der Daten abdecken
print("Anzahl der Principal Components, die 95% der Varianz der Daten abdecken:", n_components_95_variance)
# Eigenwerte (Varianzen)
print("Eigenwerte (Varianzen):")
print(pca.explained_variance_)

# Eigenvektoren (Hauptkomponenten)
print("\nEigenvektoren (Hauptkomponenten):")
print(pca.components_)

# Speichern des reduzierten Datensatzes als CSV-Datei
df_reduced.to_csv('bike_sales_reduced.csv', index=False)


# Normalisieren aller Features
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df_cleaned_encoded_no_missing)

# Erstellen eines DataFrames aus den normalisierten Daten
df_normalized = pd.DataFrame(df_normalized, columns=df_cleaned_encoded_no_missing.columns)

# Durchlaufen aller Spalten im DataFrame
for column in df_normalized.columns:
    # Überprüfen, ob die Spalte einen $ Zeichen enthält
    if '$' in df_normalized[column].astype(str):
        # Ersetzen des $ Zeichens durch das Eurozeichen
        df_normalized[column] = df_normalized[column].astype(str).str.replace('$', '€')

# Speichern des normalisierten Datensatzes als CSV-Datei
df_normalized.to_excel('bike_sales_normalized.xlsx', index=False)