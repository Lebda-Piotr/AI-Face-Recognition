import os
import cv2
from mtcnn import MTCNN

# Dla jednego pliku
def extract_face(input_image, output_dir):

    detector = MTCNN(min_face_size=100, scale_factor=0.709, steps_threshold=[0.5, 0.6, 0.7])

    os.makedirs(output_dir, exist_ok=True)

    # Wczytanie obrazu
    img = cv2.imread(input_image)
    if img is None:
        print(f"Nie udało się odczytać: {input_image}. Pomijam...")
        return

    results = detector.detect_faces(img)
    print(f"Źródło: {input_image}")
    print(f"Liczba wykrytych twarzy: {len(results)}")

    for i, result in enumerate(results):
        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)

        # Wycinanie
        face = img[y:y + h, x:x + w]

        if face.size == 0:
            print(f"Wykryta twarz ma zerowy rozmiar. Pomijam: {input_image}")
            continue

        output_path = os.path.join(output_dir, f"face_{i}_{x}_{y}.jpg")
        
        # Zapis
        success = cv2.imwrite(output_path, face)
        if not success:
            print(f"Nie udało się zapisać do: {output_path}")

# Dla wielu plików
def extract_faces(input_dir, output_dir):
    detector = MTCNN(min_face_size=100, scale_factor=0.709, steps_threshold=[0.5, 0.6, 0.7])
    for split in ["train", "val", "test"]:
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        
        if not os.path.exists(split_input_dir):
            print(f"Folder nieznaleziony: {split_input_dir}. Pomijam...")
            continue

        for category in os.listdir(split_input_dir):
            category_input_dir = os.path.join(split_input_dir, category)
            category_output_dir = os.path.join(split_output_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)

            for file_name in os.listdir(category_input_dir):
                img_path = os.path.join(category_input_dir, file_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Nie udało się odczytać: {img_path}. Pomijam...")
                    continue
                
                results = detector.detect_faces(img)
                print(f"Źródło: {img_path}")
                print(f"Liczba wykrytych twarzy: {len(results)}")
                
                for i, result in enumerate(results):
                    x, y, w, h = result['box'] 
                    x, y = max(0, x), max(0, y) 
                    
                    # Wycinanie
                    face = img[y:y+h, x:x+w]
                    
                    if face.size == 0:
                        print(f"Wykryta twarz ma zerowy rozmiar. Pomijam: {img_path}")
                        continue
                    
                    output_path = os.path.join(category_output_dir, f"{file_name.split('.')[0]}_face{i}_{x}_{y}.jpg")
                    
                    # Zapis
                    success = cv2.imwrite(output_path, face)
                    if not success:
                        print(f"Nie udało się zapisać do: {output_path}")

if __name__ == "__main__":
    extract_faces("data/processed", "data/faces")