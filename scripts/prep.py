from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def verify_directory_structure(base_dir):
    if not os.path.exists(base_dir):
        raise ValueError(f"Katalog {base_dir} nie istnieje.")
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        raise ValueError(f"Katalog {base_dir} nie zawiera podfolderów klas.")
    print(f"Znaleziono klasy w {base_dir}: {subdirs}")

def prepare_data():
    train_dir = 'data/faces/train'
    val_dir = 'data/faces/val'

    verify_directory_structure(train_dir)
    verify_directory_structure(val_dir)

    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    return train_data, val_data

if __name__ == '__main__':
    try:
        train_data, val_data = prepare_data()
        print("Liczba obrazów w zbiorze treningowym:", train_data.samples)
        print("Liczba obrazów w zbiorze walidacyjnym:", val_data.samples)
    except ValueError as e:
        print("Błąd:", e)