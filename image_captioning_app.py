# Step 1: import libraries
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Step 2: load the pretrained model
model_name = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Step 3 : Define the image captioning function
def caption_image(input_image: np.ndarray):
    # convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # process the image
    inputs = processor(images=raw_image, return_tensors="pt")

    # generate a caption for the image
    outputs = model.generate(**inputs)

    # decode the generated tokens to text and store it into 'caption'
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption


# Step 4: Create the Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs='text',
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch()