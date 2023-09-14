import torchvision
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import torch

from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_urls = [
    "https://images.unsplash.com/photo-1481349518771-20055b2a7b24?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8cmFuZG9tfGVufDB8fDB8fHww&w=1000&q=80",
    "https://expertphotography.b-cdn.net/wp-content/uploads/2021/09/Types-of-Street-Photography-Brad-Starkey.2.jpg",
    "https://i.pinimg.com/736x/59/95/18/5995186a3da28eef8906f5d3878c76c2.jpg",
    "https://storage.googleapis.com/support-forums-api/attachment/thread-13759870-3113698012515591756.png",
    "https://i.guim.co.uk/img/media/1eea25e4bf729039a2e8b59b5908e8ff74cb7351/0_0_1590_1590/master/1590.jpg?width=700&quality=85&auto=format&fit=max&s=48bbc17e2c585105358efbfa4739e115",
    "https://images.unsplash.com/photo-1539651044670-315229da9d2f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fHN0cmVldHxlbnwwfHwwfHx8MA%3D%3D&w=1000&q=80"
]

for x in range(len(image_urls)):
    image = Image.open(requests.get(image_urls[x], stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    print(image_features.size())
    print(image_features)