
import gradio as gr
import numpy as np
from PIL import Image
import os
from src.utils.image import save_uploaded_file
from src.data.transform import transform_image
from src.apps.model1 import process_model1, process_modelFore, process_modelSmile, process_modelPore, process_modelPig


def main(input_image: np.ndarray):
    save_uploaded_file(input_image)

    pil_image = Image.fromarray(input_image)

    # convert input_image to tensor
    tensor_image = transform_image(pil_image).unsqueeze(0)

    print(tensor_image.shape)

    # outEye = process_model1(tensor_image)
    # outFore = process_modelFore(tensor_image)
    # outSmile = process_modelSmile(tensor_image)
    outPig = process_modelPig(tensor_image)
    # outPore = process_modelPore(tensor_image)

    result = {
        # 'Wrinkle Eye': outEye,
        # 'Wrinkle Fore': outFore,
        # 'Wrinkle Smiline': outSmile,
        'Pigmentation': outPig,
        # 'Pore': outPore
    }

    return result
