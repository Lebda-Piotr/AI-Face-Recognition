import os
import shutil
import random

def split_data(source_dir, target_dir, split_ratios=(0.7, 0.2, 0.1)):
    categories = os.listdir(source_dir)
    for category in categories:
        category_path = os.path.join(source_dir, category)
        files = os.listdir(category_path)
        random.shuffle(files)

        train_split = int(split_ratios[0] * len(files))
        val_split = int((split_ratios[0] + split_ratios[1]) * len(files))

        train_files = files[:train_split]
        val_files = files[train_split:val_split]
        test_files = files[val_split:]

        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_dir = os.path.join(target_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for file_name in split_files:
                shutil.copy(os.path.join(category_path, file_name), os.path.join(split_dir, file_name))

if __name__ == "__main__":
    random.seed(42)
    split_data("data/raw", "data/processed")