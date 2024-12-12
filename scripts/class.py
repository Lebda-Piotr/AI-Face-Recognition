import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from face import extract_face  

class_to_actor = {
    0: "Ed Harris",
    1: "Jim Carrey",
    2: "Laura Linney",
    3: "Natascha McElhone",
    4: "Noah Emmerich"
}

# Funkcja do ładowania modelu
def load_cnn_model(model_path='models/truman_classifier.h5'):
    return load_model(model_path)

# Funkcja przygotowująca obraz do klasyfikacji
def prepare_image(img, target_size=(128, 128)):
    img = cv2.resize(img, target_size)  
    img = img.astype('float32') / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def classify_faces(faces, model):
    predictions = []
    for face in faces:
        face = prepare_image(face)  
        pred = model.predict(face)  
        class_idx = np.argmax(pred, axis=1)[0]  # Wybór klasy z największym prawdopodobieństwem
        predictions.append(class_idx)  # Dodanie do listy wyników
    return predictions

def classify_faces_from_image(input_image, model):
    temp_output_dir = "temp_faces"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    extract_face(input_image, temp_output_dir)  
    
    faces = []
    # Zbieranie twarzy wyciętych 
    for file_name in os.listdir(temp_output_dir):
        img_path = os.path.join(temp_output_dir, file_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Nie udało się odczytać: {img_path}. Pomijam...")
            continue
        
        faces.append(img)  
    
    if faces:
        predictions = classify_faces(faces, model)
        actor_list = []
        for i, pred in enumerate(predictions):
            actor_name = class_to_actor.get(pred, "Nieznany aktor")  # Pobierz imię aktora lub domyślny tekst
            actor_list.append(f"{actor_name}")
        
        print("Aktorzy, których widzę na zdjęciu: " + ", ".join(actor_list))
    else:
        print("Nie wykryto twarzy w zdjęciu.")
    
    # Usuwam tymczasowy folder 
    for file_name in os.listdir(temp_output_dir):
        os.remove(os.path.join(temp_output_dir, file_name))
    os.rmdir(temp_output_dir)

if __name__ == "__main__":
    input_image = input("Proszę podać pełną ścieżkę do zdjęcia: ")
    
    if not os.path.exists(input_image):
        print("Podany plik nie istnieje. Sprawdź ścieżkę.")
        exit(1)
    
    model = load_cnn_model()  

    classify_faces_from_image(input_image, model)