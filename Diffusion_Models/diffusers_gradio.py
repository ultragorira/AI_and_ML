import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import os

def get_token() -> str:
  return os.environ.get("HUGGING_FACE_TOKEN") 

def save_images(images: list) -> list:

  output_files_names = []
  for id, image in enumerate(images):
    filename = f"output{id}.png"
    image.save(filename)
    output_files_names.append(filename)
    
  return output_files_names



def create_img(prompt :str, number_output_requested: int) -> list:
  AUTH_TOKEN = get_token()
  generator = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                        revision="fp16", 
                                                        torch_dtype=torch.float16, 
                                                        use_auth_token=AUTH_TOKEN)
  generator.to("cuda")
  prompt = [prompt] * number_output_requested
  with torch.autocast("cuda"):
    images = generator(prompt).images
    output_paths = save_images(images)
  return output_paths

diffusers_app = gr.Interface(
        fn=create_img,
        inputs =
        [
          gr.Textbox(label="Write your prompt below", placeholder = "A squirrel bench pressing 200 kg"),
          gr.Slider(value=1, minimum=1, maximum=8, step=1, label="Number of pictures to generate")
        ],
        outputs = gr.Gallery(label="Generated Images").style(grid=[2]),
        title="Text to Image with Stable Diffusion",
        description="This is a basic app to generate pictures with Stable Diffusion."
) 

diffusers_app.launch(debug=True)