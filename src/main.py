
import gradio as gr
import numpy as np
from PIL import Image
import os
from src.utils.image import save_uploaded_file
from src.data.transform import transform_image
from src.apps.model1 import process_model1


def main(input_image: np.ndarray):
    save_uploaded_file(input_image)

    pil_image = Image.fromarray(input_image)

    # convert input_image to tensor
    tensor_image = transform_image(pil_image).unsqueeze(0)

    print(tensor_image.shape)

    out1 = process_model1(tensor_image)

    result = {
        'model1': out1
    }

    return result
