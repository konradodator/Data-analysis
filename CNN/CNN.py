import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Funktion zum Laden und Vorbereiten von Bildern
def bilder_laden(dataset_verzeichnis):
    bilddateien = []
    labels = []

    for root, dirs, files in os.walk(dataset_verzeichnis):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                bildpfad = os.path.join(root, file)
                bilddateien.append(bildpfad)
                # Label aus dem Ordnernamen extrahieren
                label = os.path.basename(os.path.dirname(bildpfad))
                labels.append(label)

    return bilddateien, labels


# Funktion für Labels
def labels_codieren(labels):
    einzigartige_labels = np.unique(labels)
    label_zu_index = {label: i for i, label in enumerate(einzigartige_labels)}
    index_zu_label = {i: label for label, i in label_zu_index.items()}

    codierte_labels = [label_zu_index[label] for label in labels]

    return np.array(codierte_labels), label_zu_index, index_zu_label


# Funktion zur Vorverarbeitung von Bildern
def bilder_vorverarbeiten(bilddateien, labels, img_groesse=(150, 150)):
    bilder = []
    for bildpfad in bilddateien:
        img = cv2.imread(bildpfad)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # In RGB konvertieren
        img = cv2.resize(img, img_groesse)  # Größe ändern
        img = img / 255.0  # Normalisieren
        bilder.append(img)

    return np.array(bilder), np.array(labels)


# Laden und Vorverarbeiten von Bildern
dataset_verzeichnis = "cats_vs_dogs"
bilddateien, labels = bilder_laden(dataset_verzeichnis)
codierte_labels, label_zu_index, index_zu_label = labels_codieren(labels)
bilder, labels = bilder_vorverarbeiten(bilddateien, codierte_labels)

# Aufteilen des Datensatzes in Trainings- und Testsets
trainings_bilder, test_bilder, trainings_labels, test_labels = train_test_split(bilder, labels, test_size=0.2,
                                                                                random_state=42)

# CNN-Architektur
modell = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Modell kompilieren
modell.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

# Hyperparameter definieren
batch_groesse = 32
epochen = 30

# Listen zum Speichern von Loss und Accuracy
loss_verlauf = []
accuracy_verlauf = []
beste_val_loss = np.inf
beste_val_accuracy = 0.0
bestes_modell_dateiname = "bestes_modell.h5"

# Trainingsloop
for epoche in range(epochen):
    print(f"Epoche {epoche + 1}/{epochen}")
    # Trainingsdatensatz mischen
    indices = np.arange(len(trainings_bilder))
    np.random.shuffle(indices)
    trainings_bilder = trainings_bilder[indices]
    trainings_labels = trainings_labels[indices]

    # Batch-Training
    for i in range(0, len(trainings_bilder), batch_groesse):
        batch_bilder = trainings_bilder[i:i + batch_groesse]
        batch_labels = trainings_labels[i:i + batch_groesse]
        modell.train_on_batch(batch_bilder, batch_labels)

    # Bewertung auf Testdatensatz
    verlust, genauigkeit = modell.evaluate(test_bilder, test_labels)
    print(f"Test Verlust: {verlust}, Test Genauigkeit: {genauigkeit}")

    # Loss und Accuracy zum Verlauf hinzufügen
    loss_verlauf.append(verlust)
    accuracy_verlauf.append(genauigkeit)

    # Modell speichern, wenn die Loss verbessert wurde und die Genauigkeit höher ist
    if verlust < beste_val_loss and genauigkeit > beste_val_accuracy:
        beste_val_loss = verlust
        beste_val_accuracy = genauigkeit
        bestes_modell_dateiname = f"bestes_modell_loss_{verlust:.4f}_acc_{genauigkeit:.4f}.h5"
        modell.save(bestes_modell_dateiname)
        print(f"Das Modell wurde gespeichert: {bestes_modell_dateiname}")

    # Break, wenn Verbesserung an die angegebenen Grenzen kommt
    if epoche > 0 and verlust < 0.3 and genauigkeit > 0.98:
        print(f"Keine Verbesserung mehr in Epoche {epoche + 1}. Beende Training.")
        break

# Plotten der Loss-Entwicklung
plt.figure(figsize=(10, 5))
plt.plot(loss_verlauf, label='Loss', color='red')
plt.xlabel('Epochen')
plt.ylabel('Loss')
plt.title('Entwicklung des Loss während des Trainings')
plt.legend()
plt.grid(True)
plt.savefig("loss_verlauf.png")  # Loss-Verlauf als Bild speichern
plt.show()

# Plotten der Accuracy-Entwicklung
plt.figure(figsize=(10, 5))
plt.plot(accuracy_verlauf, label='Accuracy', color='blue')
plt.xlabel('Epochen')
plt.ylabel('Accuracy')
plt.title('Entwicklung der Accuracy während des Trainings')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_verlauf.png")  # Accuracy-Verlauf als Bild speichern
plt.show()



print(f"Das Modell mit der besten Genauigkeit wurde als '{bestes_modell_dateiname}' gespeichert.")