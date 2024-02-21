import gradio as gr
import numpy as np
from PIL import Image
import os
from src.main import main


with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "### <center> Skin Analyst </center>")

            input_image = gr.Image(type="numpy")

            start_button = gr.Button(
                "Start", elem_id="start-btn", visible=True)
        with gr.Column():
            gr.Markdown(
                "### <center> Skin Analyst Results </center>")

            with gr.Row(variant='panel'):
                skin_result = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "### <center> Face recognition </center>")

            image_face_recognition = gr.Image(type="numpy")

        with gr.Column():
            gr.Markdown(
                "### <center> Face crop </center>")

            input_image_face_crop = gr.Image(type="numpy")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "### <center> Fore Crop </center>")

            input_image_fore_crop = gr.Image(type="numpy")
        with gr.Column():
            gr.Markdown(
                "### <center> Eye Crop </center>")

            input_image_eye_crop = gr.Image(type="numpy")
        with gr.Column():
            gr.Markdown(
                "### <center> Smile Crop </center>")

            input_image_smile_crop = gr.Image(type="numpy")
    with gr.Row():
        # markdown output

        start_button.click(main, [input_image], outputs=[
            image_face_recognition, input_image_face_crop, input_image_fore_crop, input_image_eye_crop, input_image_smile_crop, skin_result])

demo.launch(debug=True, show_api=True)
