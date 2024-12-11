import os
import cv2
from mtcnn import MTCNN

def extract_faces(input_dir, output_dir):
    detector = MTCNN()
    for split in ["train", "val", "test"]:
        split_input_dir = os.path.join(input_dir, "Truman", split)
        split_output_dir = os.path.join(output_dir, "Truman", split)
        os.makedirs(split_output_dir, exist_ok=True)

        if not os.path.exists(split_input_dir):
            print(f"Folder nieznaleziony: {split_input_dir}. Pomijam...")
            continue

        for file_name in os.listdir(split_input_dir):
            img_path = os.path.join(split_input_dir, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Nie udało się odcytać: {img_path}. Pomijam...")
                continue
            
            results = detector.detect_faces(img)
            print(f"Źródło: {img_path}")
            print(f"Liczba wykrytych twarzy: {len(results)}")
            
            for i, result in enumerate(results):
                x, y, w, h = result['box'] 
                x, y = max(0, x), max(0, y) 
                
                # Wycinanie
                face = img[y:y+h, x:x+w]
                
                # Sprawdzenie rozmiaru
                if face.size == 0:
                    print(f"Wykryta twarz ma zerowy rozmiar. Pomijam: {img_path}")
                    continue
                
                # Generowanie nazwy pliku
                output_path = os.path.join(split_output_dir, f"{file_name.split('.')[0]}_face{i}_{x}_{y}.jpg")
                
                # Zapis
                success = cv2.imwrite(output_path, face)
                if not success:
                    print(f"Nie udało się zapisać do: {output_path}")

if __name__ == "__main__":
    extract_faces("data/processed", "data/faces")