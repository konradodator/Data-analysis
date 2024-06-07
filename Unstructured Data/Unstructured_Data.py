import os
import cv2
import matplotlib.pyplot as plt

# Pfad zum Datensatz-Ordner
dataset_path = 'Agricultural-crops'

# Liste zum Speichern aller Bilddateien
image_files = []
# Liste zum Speichern der Anzahl der Bilder pro Klasse (Unterordner)
class_counts = {}
# Liste zum Speichern der Aspect Ratios und der Größen
aspect_ratios = []
image_sizes = []
# Durchlaufen aller Unterordner und Sammeln der Bilddateien
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(root, file)
            image_files.append(image_path)
            # Name des Unterordners als Klasse
            class_name = os.path.basename(root)
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            # Bild laden und Aspect Ratio und Größe berechnen
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            aspect_ratio = width / height
            size = width * height
           #Daten speichern
            aspect_ratios.append(aspect_ratio)
            image_sizes.append(size)


# Sortieren der Dateien
image_files.sort()

# Initialisieren des Index
current_index = 0


#  Anzeigen eines Bildes mit Label
def show_image(index):
    global image_files
    # Bilddatei-Pfad
    img_path = image_files[index]
    # Laden des Bildes
    img = cv2.imread(img_path)

    # Label aus dem Dateinamen und dem Unterordner
    folder_name = os.path.basename(os.path.dirname(img_path))
    file_name = os.path.basename(img_path).rsplit('.', 1)[0]
    label = f"{folder_name} - {file_name}"

    # Bildgröße anpassen für eine bessere Anzeige
    resized_img = cv2.resize(img, (800, 600))

    # Label auf das Bild schreiben
    cv2.putText(resized_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Bild anzeigen
    cv2.imshow('Image Viewer', resized_img)


# Bild anzeigen
show_image(current_index)



# Funktion zum Steuern der Navigation durch Tastendruck
def navigate_images():
    global current_index
    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC-Taste zum Beenden
            break
        elif key == ord('a'):  # 'a' Taste für links
            current_index = (current_index - 1) % len(image_files)
            show_image(current_index)
        elif key == ord('d'):  # 'd' Taste für rechts
            current_index = (current_index + 1) % len(image_files)
            show_image(current_index)
        elif key == ord('p'):  # 'p' Taste zum Plotten von 10 ausgewählten Bildernund Speichern
            plot_images()
        elif key == ord('c'):  # 'c' Taste zum Plotten der Anzahl der Bilder pro Klasse
            plot_class_distribution()
        elif key == ord('r'):  # 'r' Taste zum Plotten der Aspect Ratio Verteilung
            plot_aspect_ratio_distribution()
        elif key == ord('s'):  # 's' Taste zum Plotten der Bildgröße  Verteilung
            plot_image_size_distribution()

# Funktion zum Plotten und Speichern einer Auswahl von 10 Bildern aus verschiedenen Ordnern
def plot_images():
    # Liste zum Speichern der Bildauswahl pro Klasse
    selected_images = {}

    # Erste 10 Bilder aus verschiedenen Klassen auswählen
    for img_path in image_files:
        class_name = os.path.basename(os.path.dirname(img_path))
        if class_name not in selected_images:
            selected_images[class_name] = img_path
        if len(selected_images) == 10:
            break

    # Initialisieren des Plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Anzeigen und Beschriften der ausgewählten Bilder
    for idx, (class_name, img_path) in enumerate(selected_images.items()):
        # Bild laden
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konvertieren von BGR nach RGB für Matplotlib

        # Label aus dem Dateinamen und dem Unterordner extrahieren
        file_name = os.path.basename(img_path).rsplit('.', 1)[0]
        label = f"{class_name} - {file_name}"

        # Bild in der entsprechenden Achse anzeigen
        ax = axes[idx // 5, idx % 5]
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')

    # Layout anpassen
    plt.tight_layout()

    # Plot speichern
    output_path = 'image_grid.png'
    plt.savefig(output_path)
    print(f"Plot gespeichert als {output_path}")

    # Plot anzeigen
    plt.show()


# Funktion zum Plotten der Anzahl der Bilder pro Klasse
def plot_class_distribution():
    # Daten für das Plotten vorbereiten
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Plotten der Anzahl der Bilder pro Klasse
    plt.figure(figsize=(12, 8))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Klassen')
    plt.ylabel('Anzahl der Bilder')
    plt.title('Anzahl der Bilder pro Klasse')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Plot speichern
    plot_path = 'class_distribution.png'
    plt.savefig(plot_path)
    print(f"Plot gespeichert als {plot_path}")

    # Plot anzeigen
    plt.show()


# Funktion zum Plotten der Aspect Ratio Verteilung
def plot_aspect_ratio_distribution():
    plt.figure(figsize=(12, 8))
    plt.hist(aspect_ratios, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Anzahl der Bilder')
    plt.title('Verteilung der Aspect Ratio')
    plt.tight_layout()

    # Plot speichern
    plot_path = 'aspect_ratio_distribution.png'
    plt.savefig(plot_path)
    print(f"Plot gespeichert als {plot_path}")

    # Plot anzeigen
    plt.show()


# Funktion zum Plotten der Bildgrößen Verteilung
def plot_image_size_distribution():
    plt.figure(figsize=(12, 8))
    plt.hist(image_sizes, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Bildgröße (Pixel^2)')
    plt.ylabel('Anzahl der Bilder')
    plt.title('Verteilung der Bildgrößen')
    plt.tight_layout()

    # Plot speichern
    plot_path = 'image_size_distribution.png'
    plt.savefig(plot_path)
    print(f"Plot gespeichert als {plot_path}")

    # Plot anzeigen
    plt.show()


# Funktion zum Anpassen eines Bildes mit/ohne Letterboxing
def resize_image(img, target_size=(256, 256)):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if 0.8 <= aspect_ratio <= 1.2:
        # Resize ohne Letterboxing
        resized_img = img.copy()
    else:
        # Resize mit Letterboxing
        new_width = target_size[0]
        new_height = target_size[1]

        if aspect_ratio > 1.2:
            scale = new_width / width
            new_height = int(height * scale)
            resized_img = cv2.resize(img, (new_width, new_height))
            top_bottom_padding = (target_size[1] - new_height) // 2
            resized_img = cv2.copyMakeBorder(resized_img, top_bottom_padding, top_bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            scale = new_height / height
            new_width = int(width * scale)
            resized_img = cv2.resize(img, (new_width, new_height))
            left_right_padding = (target_size[0] - new_width) // 2
            resized_img = cv2.copyMakeBorder(resized_img, 0, 0, left_right_padding, left_right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # Normalisierung der Pixelwerte
            resized_img = resized_img / 255.0

    return resized_img


output_path = 'Resized-Images-Normalized'
os.makedirs(output_path, exist_ok=True)

# Alle Bilder resizen und normieren, bevor sie gespeichert werden
for img_path in image_files:
    img = cv2.imread(img_path)
    resized_img = resize_image(img)  # Die bearbeitete Version des Bildes erhalten
    resized_normalized_img = resized_img / 255.0  # Normalisierung der Pixelwerte

    # Ziel Pfad beibehalten und Dateiendung zu PNG ändern
    relative_path = os.path.relpath(img_path, dataset_path)
    save_path = os.path.join(output_path, os.path.splitext(relative_path)[0] + '.PNG')

    # Ordnerstruktur erstellen
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Bild als PNG speichern
    cv2.imwrite(save_path, resized_normalized_img * 255,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Navigationsfunktion starten
navigate_images()

# Fenster schließen
cv2.destroyAllWindows()
