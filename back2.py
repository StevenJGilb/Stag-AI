import gradio as gr
from diffusers import StableDiffusionLDM3DPipeline

model_id = "Intel/ldm3d-pano"
pipe = StableDiffusionLDM3DPipeline.from_pretrained(model_id)
pipe.to("cpu")


# This function will be called when the user submits their choices
def design_room(room_type, style):
    # For now, we'll just return the choices. Later, this function will create the room design.
    prompt = f"Add furniture, decorations and design my {room_type} in {style} style."
    name = "bedroom_pano"
    output = pipe(prompt, width=1024, height=512, guidance_scale=7.0, num_inference_steps=50) 
    
    rgb_image, depth_image = output.rgb, output.depth
    rgb_image[0].save(name+"_ldm3d_rgb.jpg")
    depth_image[0].save(name+"_ldm3d_depth.png")

    return prompt

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
    output = gr.Textbox()

    # When the button is clicked, call the design_room function
    submit_button.click(fn=design_room, inputs=[room_type, style], outputs=output)

# Launch the application
demo.launch()