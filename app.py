import gradio as gr
from inference_engine import PunctuationRestorer
import os

# Initialize the model (global scope for efficiency)
MODEL_PATH = "."

restorer = None

if os.path.exists(MODEL_PATH):
    restorer = PunctuationRestorer(MODEL_PATH)
else:
    print("Model not found.")


def restore_punctuation(text: str) -> str:
    """Restore punctuation and capitalization to input text."""
    if restorer is None:
        return "Error: Model not loaded. Please train the model first."
    
    if not text or not text.strip():
        return "Please enter some text."
    
    try:
        result = restorer.restore(text)
        return result
    except Exception as e:
        return f"Error during restoration: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Punctuation Restoration") as demo:
    gr.Markdown("# Punctuation and Case Restoration")
    gr.Markdown("Enter unpunctuated text (e.g., from ASR output) and get properly punctuated text.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text (unpunctuated)",
                placeholder="enter your text here without punctuation all lowercase",
                lines=10
            )
            submit_btn = gr.Button("Restore Punctuation", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Restored Text",
                lines=10
            )
    
    gr.Examples(
        examples=[
            ["hello how are you doing today"],
            ["what time is the meeting tomorrow can you let me know"],
            ["this is amazing i cant believe it works so well"],
        ],
        inputs=input_text
    )
    
    submit_btn.click(
        fn=restore_punctuation,
        inputs=input_text,
        outputs=output_text
    )


if __name__ == "__main__":
    demo.launch(share=False)
