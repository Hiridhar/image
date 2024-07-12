import os
import streamlit as st
from PIL import Image
import numpy as np
import albumentations as A
from io import BytesIO
from zipfile import ZipFile

UPLOAD_FOLDER = 'uploads'
AUGMENTED_FOLDER = 'augmented'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

def augment_image(image_path):
    image = np.array(Image.open(image_path))
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=40, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(scale_limit=0.1, p=0.5)
    ])
    augmented_images = [transform(image=image)['image'] for _ in range(5)]
    return augmented_images

def save_augmented_images(images, original_filename):
    augmented_paths = []
    for i, aug_image in enumerate(images):
        aug_image_pil = Image.fromarray(aug_image)
        aug_filename = f"{original_filename.split('.')[0]}_aug_{i}.png"
        aug_path = os.path.join(AUGMENTED_FOLDER, aug_filename)
        aug_image_pil.save(aug_path)
        augmented_paths.append(aug_path)
    return augmented_paths

def zip_augmented_images(image_paths):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        for image_path in image_paths:
            zip_file.write(image_path, arcname=os.path.basename(image_path))
    zip_buffer.seek(0)
    return zip_buffer

st.title("Image Augmentation Application")
st.write("Upload images to generate augmented data and download the results as a ZIP file.")

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}"):
            # Save uploaded file to disk
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Augment the image
            augmented_images = augment_image(file_path)
            
            # Save augmented images
            augmented_paths = save_augmented_images(augmented_images, uploaded_file.name)
    
    # Create ZIP file of augmented images
    zip_buffer = zip_augmented_images(augmented_paths)
    
    st.success("Augmentation complete! Download your augmented images below.")
    st.download_button(
        label="Download ZIP",
        data=zip_buffer,
        file_name="augmented_images.zip",
        mime="application/zip"
    )
