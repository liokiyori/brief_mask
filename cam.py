import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Chargez le modèle pré-entraîné
model = load_model("best_model.h5")

# Démarrez la capture vidéo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Retournez horizontalement l'image
    frame = cv2.flip(frame, 1)

    # Redimensionnez l'image
    img = cv2.resize(frame, (224, 224))

    # Prétraitez l'image
    img = img / 255.0

    # Passez l'image au modèle pour obtenir une prédiction
    prediction = model.predict(np.expand_dims(img, axis=0))[0]

    # Obtenez l'indice de la classe avec la probabilité maximale
    predicted_class_index = np.argmax(prediction)

    # Déterminez la classe prédite en fonction de l'indice
    if predicted_class_index == 0:
        label = "Sans Masque"
    else:
        label = "Avec Masque"

    # Affichez sur l'image
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Affichez la vidéo
    cv2.imshow("Webcam", frame)

    # Quittez q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérez les ressources
cap.release()
cv2.destroyAllWindows()
