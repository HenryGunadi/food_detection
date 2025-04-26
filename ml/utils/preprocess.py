import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

data_transform = transforms.Compose([
    # resize image
    transforms.Resize(size=(64, 64)),

    # flip images
    transforms.RandomHorizontalFlip(p=0.5),

    # turn image to torch tensor
    transforms.ToTensor()
])

def plot_transformed_images(image_path: Path, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_path, n=n)

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
