import os
from typing import LiteralString
from PIL import Image
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List

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

def plot_eval_curves(results: dict[str, List[float]]):
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]
    
    train_acc = results["train_acc"]
    val_acc = results["val_acc"]
    
    epochs = range(len(train_loss))

    plt.figure(figsize=(15, 7))

    # Plot train vs val loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot train vs val acc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()