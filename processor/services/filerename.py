
import os
from pathlib import Path
import uuid

# Enter your folder path here
FOLDER_PATH = r"D:\Development\Image LabelerBgremove\test1"

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".gif", ".tif", ".tiff"
}

def rename_images(folder_path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Invalid folder path: {folder}")
        return

    # Get image files only, sorted by filename
    image_files = sorted(
        [
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda file: file.name.lower()
    )

    if not image_files:
        print("No image files found.")
        return

    print(f"Found {len(image_files)} images.")

    # Step 1: Temporary names prevent filename conflicts
    temp_files = []

    for file in image_files:
        temp_name = folder / f"__temp_{uuid.uuid4().hex}{file.suffix.lower()}"
        file.rename(temp_name)
        temp_files.append((temp_name, file.suffix.lower()))

    # Step 2: Rename images sequentially
    for number, (temp_file, extension) in enumerate(temp_files, start=1):
        new_name = folder / f"{number}{extension}"
        temp_file.rename(new_name)
        print(f"{temp_file.name} -> {new_name.name}")

    print(f"\nSuccessfully renamed {len(image_files)} images.")

if __name__ == "__main__":
    rename_images(FOLDER_PATH)
