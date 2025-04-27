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

def plot_transformed_image(image_path: Path, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_path, k=n)

    for image_path in random_image_paths:
        with Image.open(image_path) as img:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img)
            ax[0].set_title(f"Original \nSize {img.size}")
            ax[0].axis("off")

            transformed_image = transform(img).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize : {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class : {image_path.parent.stem}", fontsize=16)