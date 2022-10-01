import gradio as gr
from diffusers import DiffusionPipeline
import os

def get_token() -> str:
    return os.environ.get("HUGGING_FACE_TOKEN") 

def create_img(prompt :str) -> str:
    output_path = "output.png"
    AUTH_TOKEN = get_token()
    generator = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", use_auth_token=AUTH_TOKEN)
    generator.to("cuda")
    image = generator(prompt).images[0]
    image.save(output_path)
    return output_path

diffusers_app = gr.Interface(
        fn=create_img,
        inputs="text",
        outputs="image"
) 

diffusers_app.launch(debug=True)