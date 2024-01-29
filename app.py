import gradio as gr
import numpy as np
from PIL import Image
import os
from src.main import main

iface = gr.Interface(
    fn=main,
    inputs=gr.Image(type="numpy"),  # Use Sketchpad as input
    outputs=[
        gr.Image(type="numpy", label="Face recognition"),
        gr.Image(type="numpy", label="Face crop"),
        gr.Image(type="numpy", label="Fore Crop"),
        gr.Image(type="numpy", label="Eye Crop"),
        gr.Image(type="numpy", label="Smile Crop"),
        gr.Textbox(lines=10, label="Results")
    ],  # Output the image
    live=True,
)

iface.launch(server_port=8080)
