import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token="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")
pipe = pipe.to("cuda")

def design_room(room_type, style):
    # For now, we'll just return the choices. Later, this function will create the room design.
    prompt = f"Add furniture, decorations and design my {room_type} in {style} style."
    #name = f"{room_type}"
    output = pipe(prompt, size="256x256").images[0]  
    
    #output.save(f"{room_type}_{style}.png")
    return output

# Define the options for room types and styles
room_types = ["Offices", "Bedrooms", "Living Rooms", "Kitchens"]
styles = ["Rustic", "Indian", "European", "American", "Chinese"]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Room Design Application")
    with gr.Row():
        room_type = gr.Dropdown(label="Select Room Type", choices=room_types)
        style = gr.Dropdown(label="Select Style", choices=styles)
    submit_button = gr.Button("Design Room")
    output = gr.Image()

    # When the button is clicked, call the design_room function
    submit_button.click(fn=design_room, inputs=[room_type, style], outputs=output)

# Launch the application
demo.launch()
