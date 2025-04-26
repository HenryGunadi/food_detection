import os
from typing import LiteralString
from PIL import Image
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_random_image(image_path: Path):
    image_path_list = list(image_path.rglob("*.jpeg"))

    random_img_path = random.choice(image_path_list)
    image_class = random_img_path.parent.stem
    img = Image.open(random_img_path) 
    img_as_array = np.array(img)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_as_array)
    plt.title(f"Image class : {image_class} | Image shape : {img_as_array.shape} -> [height, width, color channel]")
    plt.axis(False)

def test():
    print("Hello")