import gradio as gr
import numpy as np
from PIL import Image
import os
from src.main import main

iface = gr.Interface(
    fn=main,
    inputs=gr.Image(type="numpy"),  # Use Sketchpad as input
    outputs=gr.Textbox(lines=10, label="Results"),  # Output the image
    live=True,
)

iface.launch(server_port=7860)
