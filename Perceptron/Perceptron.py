import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Zufallsgenerator für Reproduzierbarkeit setzen
np.random.seed(42)

# Anzahl der Datenpunkte
n_samples = 1000

# Zwei Cluster für zwei Klassen erzeugen
mean_class_0 = [2, 2]
cov_class_0 = [[1, 0.5], [0.5, 1]]

mean_class_1 = [6, 6]
cov_class_1 = [[1, 0.5], [0.5, 1]]

# Zufallsdaten für beide Klassen erzeugen
class_0_data = np.random.multivariate_normal(mean_class_0, cov_class_0, n_samples // 2)
class_1_data = np.random.multivariate_normal(mean_class_1, cov_class_1, n_samples // 2)

# Labels erzeugen
class_0_labels = np.zeros(n_samples // 2)
class_1_labels = np.ones(n_samples // 2)

# Daten und Labels kombinieren
data = np.vstack((class_0_data, class_1_data))
labels = np.hstack((class_0_labels, class_1_labels))

# Daten und Labels mischen
indices = np.arange(n_samples)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Datensatz plotten
plt.figure(figsize=(10, 6))
plt.scatter(data[labels == 0][:, 0], data[labels == 0][:, 1], c='blue', label='Klasse 0')
plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], c='red', label='Klasse 1')
plt.xlabel('Merkmal 1')
plt.ylabel('Merkmal 2')
plt.title('Datensatz vor dem Training')
plt.legend()
plt.savefig('dataset_before_training.png')
plt.show()

# Neuronales Netzwerk definieren
inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Modell kompilieren
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Benutzerdefinierte Trainingsschleife
epochs = 50
batch_size = 32
best_accuracy = 0.0
best_model_path = 'best_model.h5'

losses = []
accuracies = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Datensatz zu Beginn jeder Epoche mischen
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    epoch_loss = 0.0
    epoch_accuracy = 0.0
    num_batches = n_samples // batch_size

    for batch in range(num_batches):
        # Batch auswählen
        batch_data = data[batch * batch_size:(batch + 1) * batch_size]
        batch_labels = labels[batch * batch_size:(batch + 1) * batch_size]
        batch_labels = np.expand_dims(batch_labels, axis=-1)

        # Trainingsschritt
        with tf.GradientTape() as tape:
            predictions = model(batch_data, training=True)
            loss = tf.keras.losses.binary_crossentropy(batch_labels, predictions)
            epoch_loss += tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Genauigkeit berechnen
        accuracy = tf.keras.metrics.BinaryAccuracy()
        accuracy.update_state(batch_labels, predictions)
        epoch_accuracy += accuracy.result().numpy()

    # Durchschnittlicher Verlust und Genauigkeit für die Epoche berechnen
    epoch_loss /= num_batches
    epoch_accuracy /= num_batches
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

    print(f"Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

    # Überprüfen, ob das aktuelle Modell das beste ist
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        model.save(best_model_path)
        print("Best model saved!")

# Verlust und Genauigkeit während des Trainings plotten
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Genauigkeit')
plt.xlabel('Epoche')
plt.ylabel('Genauigkeit')

plt.savefig('training_metrics.png')
plt.show()

# Bestes Modell laden
best_model = tf.keras.models.load_model(best_model_path)

# Gelernte Entscheidungsgrenze plotten
plt.figure(figsize=(10, 6))
plt.scatter(data[labels == 0][:, 0], data[labels == 0][:, 1], c='blue', label='Klasse 0')
plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], c='red', label='Klasse 1')
plt.xlabel('Merkmal 1')
plt.ylabel('Merkmal 2')
plt.title('Gelernte Entscheidungsgrenze')

# Gewichte der ersten Schicht des Modells abrufen
weights_first_layer = best_model.get_weights()[0]

# Steigung und y-Achsenabschnitt der Entscheidungsgrenze berechnen
slope = -weights_first_layer[0][0] / weights_first_layer[1][0]
intercept = -weights_first_layer[1][0] / weights_first_layer[1][1]

# Entscheidungsgrenze plotten
x_boundary = np.linspace(0, 8, 100)
y_boundary = slope * x_boundary + intercept
plt.plot(x_boundary, y_boundary, c='green', label='Entscheidungsgrenze')

plt.legend()
plt.savefig('decision_boundary.png')
plt.show()

# Regression ------------------------------------------------------------------------------------------------------

# Datensatz für Regression erzeugen
X = np.random.uniform(low=0, high=10, size=(1000, 1))
y = 3 * X + 2 + np.random.normal(loc=0, scale=1, size=(1000, 1))

# Daten plotten
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Daten')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Datenverteilung')
plt.legend()
plt.show()

# Datensatz als CSV speichern
data = np.concatenate((X, y), axis=1)
df = pd.DataFrame(data, columns=['X', 'y'])
df.to_csv('regression_dataset.csv', index=False)

# Neuronales Netzwerk für Regression definieren
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Modell kompilieren
model.compile(optimizer='adam', loss='mean_squared_error')

# Benutzerdefinierte Trainingsschleife
epochs = 50
batch_size = 32

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Datensatz zu Beginn jeder Epoche mischen
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    epoch_loss = 0.0
    num_batches = len(X) // batch_size

    for batch in range(num_batches):
        # Batch auswählen
        batch_X = X[batch * batch_size:(batch + 1) * batch_size]
        batch_y = y[batch * batch_size:(batch + 1) * batch_size]

        # Trainingsschritt
        with tf.GradientTape() as tape:
            predictions = model(batch_X, training=True)
            loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(batch_y, predictions))
            epoch_loss += loss

        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Durchschnittlicher Verlust für die Epoche berechnen
    epoch_loss /= num_batches
    print(f"Loss: {epoch_loss}")

# Gelernte Kurve und Datensatz plotten
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Daten')
plt.plot(X, model.predict(X), color='red', label='Gelernte Kurve')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gelernte Kurve und Datensatz')
plt.legend()
plt.savefig('learned_curve.png')
plt.show()
