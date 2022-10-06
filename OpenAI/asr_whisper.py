import gradio as gr
from pytube import YouTube
import whisper


def transcribe_audio(model_selected,audio_input):

  model = whisper.load_model(model_selected)
  audio_input = whisper.load_audio(audio_input)
  audio_input = whisper.pad_or_trim(audio_input)
    
  mel = whisper.log_mel_spectrogram(audio_input).to(model.device)
    
  transcript_options = whisper.DecodingOptions(task="transcribe", fp16 = False)
  transcription = whisper.decode(model, mel, transcript_options)
  return transcription.text

with gr.Blocks() as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 90%; margin: 0 auto;">
              <div>
                <h1>Whisper App</h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 100%">
                Try Open AI Whisper with a recorded/uploaded audio or a link to YouTube video
              </p>
            </div>
        """
    )
    with gr.Row():
        with gr.Accordion(label="Whisper model selection"):
                with gr.Row():
                    model_selection_radio = gr.Radio(['base','small', 'medium', 'large'], value='base', interactive=True, label="Model")
    with gr.Tab("Record Prompt"):
      with gr.Row():
        recorded_audio_input = gr.Audio(source="microphone", type="filepath", label="Record your prompt to feed to Stable Diffusion!")
        audio_transcribe_btn = gr.Button("Transcribe")
      with gr.Row():
        transcribed_output_box = gr.TextArea(interactive=False, label="Transcription", placeholder="Transcription will appear here")
    with gr.Tab("Transcribe from YT"):
      with gr.Row():
        gr.TextArea()
    #####################################################    
    audio_transcribe_btn.click(transcribe_audio,
                              inputs=[
                                        model_selection_radio,
                                        recorded_audio_input
                              ],
                              outputs=transcribed_output_box
                              )

demo.launch(enable_queue=True, debug=True)