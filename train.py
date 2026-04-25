# Import potrzebnych bibliotek
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Ustawienia podstawowe
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 30

# Ścieżki do danych
train_dir = "data/train"
test_dir = "data/test"

# Przygotowanie danych treningowych z augmentacją
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Przygotowanie danych testowych bez augmentacji
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Wczytanie danych treningowych
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Wczytanie danych testowych
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Utworzenie modelu CNN
model = models.Sequential([
    # Warstwa wejściowa
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    # Pierwszy blok konwolucyjny
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Drugi blok konwolucyjny
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Trzeci blok konwolucyjny
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Zamiana danych na wektor
    layers.Flatten(),

    # Warstwa decyzyjna
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Warstwa wyjściowa dla 7 emocji
    layers.Dense(7, activation="softmax")
])

# Konfiguracja uczenia modelu
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Wyświetlenie struktury modelu
model.summary()

# Trenowanie modelu
model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# Zapis modelu do pliku
model.save("model_emotions.h5")

# Informacje końcowe
print("Model zapisany jako model_emotions.h5")
print("Klasy:", train_data.class_indices)