import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception

# Funktion zum Laden und Vorbereiten von Bildern
def bilder_laden(dataset_verzeichnis):
    bilddateien = []
    labels = []

    for root, dirs, files in os.walk(dataset_verzeichnis):
        for file in files:
            if file.lower().endswith('.jpg'):
                bildpfad = os.path.join(root, file)
                bilddateien.append(bildpfad)
                # Label aus dem Ordnernamen entnehmen
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

# Funktion zum Vorverarbeitung von Bildern
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
dataset_verzeichnis = "stadt_vs_land"
bilddateien, labels = bilder_laden(dataset_verzeichnis)
codierte_labels, label_zu_index, index_zu_label = labels_codieren(labels)
bilder, labels = bilder_vorverarbeiten(bilddateien, codierte_labels)

# Aufteilen des Datensatzes in Trainings- und Testsets
trainings_bilder, test_bilder, trainings_labels, test_labels = train_test_split(bilder, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
"""
# VGG16 Architektur
modell = VGG16(weights=None,  # Ohne vortrainierte Gewichte
                input_shape=(150, 150, 3),
                classes=1,  # Für binäre Klassifikation
                include_top=True)

# Modell kompilieren
modell.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
"""
# Xception Architektur  vortrainierte Gewichte
base_model = Xception(weights=None, include_top=False, input_shape=(150, 150, 3))
x = base_model.output
# Global Average Pooling Layer hinzufügen
x = GlobalAveragePooling2D()(x)
# Vollständig verbundene Schicht
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
layers.Dropout(0.5)

# Erstellen des Modells
modell = Model(inputs=base_model.input, outputs=predictions)

# Modell kompilieren
modell.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""
# VGG16 mit vortrainierten Gewichten
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling Layer hinzufügen
x = Dense(512, activation='relu')(x)  # Vollständig verbundene Schicht
predictions = Dense(2, activation='softmax')(x)  # Zwei Klassen

# Erstelle das Modell
modell = Model(inputs=base_model.input, outputs=predictions)

# Frieren Sie die Schichten des Basis-Modells ein
for layer in base_model.layers:
    layer.trainable = False

# Modell kompilieren
modell.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc
"""

# Hyperparameter definieren
batch_groesse = 32
epochen = 30

# Callbacks für Early Stopping und Modell Checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('bestes_modell.keras', monitor='val_accuracy', save_best_only=True)

# Modell trainieren
history = modell.fit(
    datagen.flow(trainings_bilder, trainings_labels, batch_size=batch_groesse),
    epochs=epochen,
    validation_data=(test_bilder, test_labels),
    callbacks=[early_stopping, model_checkpoint]
)

# Bewertung des Modells
test_verlust, test_genauigkeit = modell.evaluate(test_bilder, test_labels)
print(f"Test Verlust: {test_verlust}, Test Genauigkeit: {test_genauigkeit}")

# Vorhersagen für den Testdatensatz
test_vorhersagen = modell.predict(test_bilder)
test_vorhersagen = np.argmax(test_vorhersagen, axis=1)

# Confusion Matrix erstellen
cm = confusion_matrix(test_labels, test_vorhersagen)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Land', 'Stadt'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.show()

# Plotten der Trainingsverläufe
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochen')
plt.ylabel('Loss')
plt.title('Loss Verlauf')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochen')
plt.ylabel('Accuracy')
plt.title('Accuracy Verlauf')
plt.legend()
plt.grid(True)

plt.savefig("training_verlauf.png")
plt.show()

print("Das Modell mit der besten Genauigkeit wurde als 'bestes_modell.keras' gespeichert.")

# Live-Video-Test
def live_video_test(modell):
#Verwendung von OBS für Virtuelle Camera um auf Webcam zuzugreifen
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Bildvorverarbeitung für Live-Video
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Vorhersage
        prediction = modell.predict(img)
        class_idx = np.argmax(prediction)
        class_label = ['Land', 'Stadt'][class_idx]

        # Anzeigen der Vorhersage im Live-Feed
        cv2.putText(frame, f" {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Video', frame)
        # Beenden bei Drücken der 'q'-Taste
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Modell für Live-Video-Test laden
modell = models.load_model('bestes_modell.keras')
live_video_test(modell)
