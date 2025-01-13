import torch
import os
from PIL import Image
import numpy as np
from utils import PokemonResNet, custom_transform

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PokemonResNet(num_classes=len(next(os.walk('data/train'))[1]))  # Replace YOUR_NUM_CLASSES with the number of Pokémon classes
model.load_state_dict(torch.load("pokemon_cnn_model.pth"))
model = model.to(device)
model.eval()

# Function to predict the class of an input image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
    image = custom_transform(image).unsqueeze(0).to(device)  # Transform and add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Replace this path with the path to your image
image_path = "image.png"  # Replace with the path to your image

# Replace train_dataset.classes with the list of your Pokémon class names
class_names = sorted(next(os.walk('data/train'))[1])  # Replace with your actual classes

predicted_class = predict_image(image_path)
print(f"Predicted Pokémon: {class_names[predicted_class]}")
