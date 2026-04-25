# Import potrzebnych bibliotek
import cv2
import numpy as np
import tensorflow as tf

# Rozmiar obrazu taki sam jak przy treningu
IMG_SIZE = 48

# Nazwy emocji odpowiadające numerom klas
emotions = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model("model_emotions.h5")

# Wczytanie detektora twarzy z OpenCV
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Uruchomienie kamery
cap = cv2.VideoCapture(0)

# Sprawdzenie, czy kamera działa
if not cap.isOpened():
    print("Nie można otworzyć kamery.")
    exit()

# Główna pętla programu
while True:
    # Pobranie klatki z kamery
    ret, frame = cap.read()

    # Sprawdzenie, czy udało się pobrać obraz
    if not ret:
        print("Błąd odczytu z kamery.")
        break

    # Zamiana obrazu na skalę szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy na obrazie
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Przetwarzanie każdej wykrytej twarzy
    for (x, y, w, h) in faces:
        # Wycięcie obszaru twarzy
        face = gray[y:y+h, x:x+w]

        # Zmiana rozmiaru twarzy do 48x48
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        # Normalizacja pikseli do zakresu 0-1
        face = face.astype("float32") / 255.0

        # Dodanie wymiaru batcha
        face = np.expand_dims(face, axis=0)

        # Dodanie wymiaru kanału koloru
        face = np.expand_dims(face, axis=-1)

        # Predykcja emocji
        prediction = model.predict(face, verbose=0)

        # Pobranie numeru emocji z najwyższym wynikiem
        emotion_id = int(np.argmax(prediction))

        # Pobranie pewności predykcji
        confidence = float(np.max(prediction))

        # Przygotowanie tekstu do wyświetlenia
        emotion_text = f"{emotions[emotion_id]} {confidence:.2f}"

        # Narysowanie prostokąta wokół twarzy
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Wyświetlenie emocji nad twarzą
        cv2.putText(
            frame,
            emotion_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Wyświetlenie obrazu z kamery
    cv2.imshow("Emotion Recognition", frame)

    # Zakończenie programu po naciśnięciu klawisza q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Zwolnienie kamery
cap.release()

# Zamknięcie okien OpenCV
cv2.destroyAllWindows()