import gradio as gr
import whisper
import torch
import os
from diffusers import StableDiffusionPipeline

def get_device():
  return "cuda" if torch.cuda.is_available() else "cpu"

def get_token() -> str:
  return os.environ.get("HUGGING_FACE_TOKEN") 

def generate_images(prompt, scale, iterations, seed):
  AUTH_TOKEN = get_token()
  device = get_device()

  pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                        revision="fp16", 
                                                        torch_dtype=torch.float16, 
                                                        use_auth_token=AUTH_TOKEN)

  pipe.to(device)
  generator = torch.Generator(device).manual_seed(seed)
  images = pipe(prompt, num_inference_steps = iterations, guidance_scale = scale, generator=generator).images
  return images


def transcribe_audio(model_selected,audio_input):

  model = whisper.load_model(model_selected)
  audio_input = whisper.load_audio(audio_input)
  audio_input = whisper.pad_or_trim(audio_input)
  translation_output = ""
  prompt_for_sd = ""
    
  mel = whisper.log_mel_spectrogram(audio_input).to(model.device)

  transcript_options = whisper.DecodingOptions(task="transcribe", fp16 = False)
  transcription = whisper.decode(model, mel, transcript_options)
  prompt_for_sd = transcription.text

  if transcription.language != "en":
    translation_options = whisper.DecodingOptions(task="translate", fp16 = False)
    translation = whisper.decode(model, mel, translation_options)
    translation_output = translation.text
    prompt_for_sd = translation_output

  return transcription.text, translation_output, str(transcription.language).upper(), prompt_for_sd

with gr.Blocks() as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 90%; margin: 0 auto;">
              <div>
                <h1>Whisper App</h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 100%">
                Try Open AI Whisper with a recorded audio to generate images with Stable Diffusion!
              </p>
            </div>
        """
    )
    with gr.Row():
        with gr.Accordion(label="Whisper model selection"):
                with gr.Row():
                    model_selection_radio = gr.Radio(['base','small', 'medium', 'large'], value='medium', interactive=True, label="Model")
    with gr.Tab("Record Prompt"):
      with gr.Row():
        recorded_audio_input = gr.Audio(source="microphone", type="filepath", label="Record your prompt to feed to Stable Diffusion!")
        audio_transcribe_btn = gr.Button("Transcribe")
      with gr.Row():
        transcribed_output_box = gr.TextArea(interactive=False, label="Transcription", placeholder="Transcription will appear here")
        translated_output_box = gr.TextArea(interactive=True, label="Translated prompt")
        detected_language_box = gr.Textbox(interactive=False, label="Detected Language")
    with gr.Tab("Stable Diffusion"):
      with gr.Row():
        prompt_box = gr.TextArea(interactive=False, label="Prompt")
      with gr.Row():
        guidance_slider = gr.Slider(2, 15, value = 7, label = 'Guidence Scale'),
        iterations_slider = gr.Slider(10, 100, value = 25, step = 1, label = 'Number of Iterations'),
        seed_slider = gr.Slider(
                label = "Seed",
                minimum = 0,
                maximum = 2147483647,
                step = 1,
                randomize = True)
      with gr.Row():
        images_gallery = gr.Gallery(label="Generated Images").style(grid=[2])
      with gr.Row():
        generate_image_btn = gr.Button("Generate Images")
    #####################################################    
    audio_transcribe_btn.click(transcribe_audio,
                              inputs=[
                                        model_selection_radio,
                                        recorded_audio_input
                              ],
                              outputs=[transcribed_output_box,
                                        translated_output_box,
                                        detected_language_box,
                                        prompt_box
                                      ]
                              )


demo.launch(enable_queue=True, debug=True)