import gradio as gr
import PIL
#import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
# import matplotlib.pyplot as plt
# import io


model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
pipe.to("cpu")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


def process_image(image, prompt):
    """ Process an uploaded image with the given text prompt using the Hugging Face model.
    Input Parameters:
    - image: The uploaded image file.
    - prompt: A text prompt.

    Returns:
    - output_image: The processed image."""
    
    output_image = pipe(prompt, image=image, num_inference_steps=15, image_guidance_scale=1).images
    return output_image[0]

interface = gr.Interface(fn=process_image,
                         inputs=[gr.Image(type="pil"), gr.Textbox(lines=2, placeholder="Enter your prompt here...")],
                         outputs=gr.Image(type="pil"),
                         title="AI Home Staging",
                         description="Upload an image and enter a prompt to see the AI-transformed image.")

# Launch the interface
if __name__ == "__main__":
    interface.launch()