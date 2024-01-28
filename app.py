import gradio as gr
import numpy as np
from PIL import Image
import os
from utils.image import save_uploaded_file

def classify_drawing(drawing_image: np.ndarray):
    # return null if no drawing is provided
    # print image size
    print(drawing_image.shape)
    
    save_uploaded_file(np.array(drawing_image))
    if drawing_image is None:
        return None

    
    return "ok"


iface = gr.Interface(
    fn=classify_drawing,
    inputs=gr.Image(type="numpy"),  # Use Sketchpad as input
    outputs="text",
    live=True,
)

iface.launch(server_port=7860)
