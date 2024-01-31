
import gradio as gr
import numpy as np
from PIL import Image
import os
from src.utils.image import save_uploaded_file
from src.data.transform import transform_image
from src.apps.model1 import process_modelFore, process_modelSmile, process_modelPore, process_modelPig, process_modelEye
from src.apps.modelcropFace import process_crop_face
from src.apps.landmark import cropFore, cropEye, cropSmile


def main(input_image: np.ndarray):
    save_uploaded_file(input_image)

    img, face_crop = process_crop_face(input_image)

    print("image", img)

    pil_image = Image.fromarray(img)

    # pil_image = Image.fromarray(input_image)

    # tách vùng face
    try:
        foreCrop = cropFore(img)
        foreCropImage = Image.fromarray(foreCrop)
    except Exception as e:
        print(f"Error cropping forehead: {e}")
        foreCropImage = img

    try:
        eyeCrop = cropEye(img)
        eyeCropImage = Image.fromarray(eyeCrop)
    except Exception as e:
        print(f"Error cropping eyes: {e}")
        eyeCropImage = img

    try:
        smileCrop = cropSmile(img)
        smileCropImage = Image.fromarray(smileCrop)
    except Exception as e:
        print(f"Error cropping smile: {e}")
        smileCropImage = img
        
    # convert input_image to tensor
    tensor_image = transform_image(pil_image).unsqueeze(0)
    fore_tensor = transform_image(foreCropImage).unsqueeze(0)
    eye_tensor = transform_image(eyeCropImage).unsqueeze(0)
    smile_tensor = transform_image(smileCropImage).unsqueeze(0)

    # print(tensor_image.shape)

    outEye = process_modelEye(eye_tensor)
    outFore = process_modelFore(fore_tensor)
    outSmile = process_modelSmile(smile_tensor)
    outPig = process_modelPig(tensor_image)
    outPore = process_modelPore(tensor_image)

    result = {
        'Wrinkle Eye': outEye,
        'Wrinkle Fore': outFore,
        'Wrinkle Smiline': outSmile,
        'Pigmentation': outPig,
        'Pore': outPore
    }

    return [img, face_crop, foreCropImage, eyeCropImage, smileCropImage, result]
    
    
