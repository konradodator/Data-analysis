import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Laden des besten Modells
bestes_modell_dateiname = "bestes_modell.h5"
modell = load_model(bestes_modell_dateiname)

# Funktion zum Klassifizieren eines Bildes und Anzeigen des Ergebnisses
def klassifiziere_bild(bild):
    # Bild nach RGB konvertieren
    bild_rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
    # Bild vorverarbeiten (Größe ändern, normalisieren usw.)
    bild_vorverarbeitet = cv2.resize(bild_rgb, (150, 150)) / 255.0
    bild_vorverarbeitet = np.expand_dims(bild_vorverarbeitet, axis=0)

    # Vorhersage des Modells für das Bild
    vorhersage = modell.predict(bild_vorverarbeitet)[0][0]

    # Debugging-Ausgabe: Rohvorhersage anzeigen
    print(f"Rohvorhersage: {vorhersage}")

    # Klassen und Konfidenz definieren
    if vorhersage > 0.5:
        klasse = "Hund"
        vertrauen = vorhersage * 100
    else:
        klasse = "Katze"
        vertrauen = (1 - vorhersage) * 100

    # Ergebnis und Vertrauen in das Bild schreiben
    cv2.putText(bild, f"{klasse} ({vertrauen:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return bild

# Öffnen des Video-Streams der Webcam
video_stream = cv2.VideoCapture(1)
#------------------------------------------------------------------------------------------------------------------
# Normalerweise ist die integrierte Webcam auf Index 0 aber auf meinem Desktop-PC kann die USB-Webcam
# nicht aufgerufen werden über cv2, daher benutze ich OBS als Eingabequelle für den Video-Stream und über OBS die Webcam
#------------------------------------------------------------------------------------------------------------------

while True:
    # Erfassen eines Bildes vom Video-Stream
    ret, bild = video_stream.read()
    if not ret:
        break

    # Klassifizierung des Bildes
    klassifiziertes_bild = klassifiziere_bild(bild)

    # Anzeigen des klassifizierten Bildes
    cv2.imshow('Klassifiziertes Bild', klassifiziertes_bild)

    # Beenden des Video-Streams bei Drücken der ESC-Taste
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Freigeben des Video-Streams und Schließen aller Fenster
video_stream.release()
cv2.destroyAllWindows()



