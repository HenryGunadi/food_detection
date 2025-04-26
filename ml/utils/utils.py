import os
from typing import LiteralString
from PIL import Image
import random

root_dir = os.getcwd()

def get_random_image(path: LiteralString):
    path = os.path.join(root_dir, path)
    image = Image.open(path)

    image.show()

def show_plt_image(image: Image):
    
