from PIL import Image
import os

def save_uploaded_file(file):
    """Save uploaded file to disk"""
    Image.fromarray(file).save(f'uploads/{len(os.listdir("uploads"))}.png')