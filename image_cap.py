import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Step 1: Import libraries

# load the pretrained processor and model
model_name = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)


# Step 2: Fetch the model and initialize a tokenizer

# load the image
img_path = "dog.jpeg"
# convert it into an RGB format
image = Image.open(img_path).convert("RGB")

# text for conditional captioning
text = "the image is of"
inputs = processor(images=image, text=text, return_tensors='pt')

# generate a caption for image
outputs = model.generate(**inputs, max_length=50)

# decode the generate tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)