import torch
import os
from pathlib import Path

# y = 0
# x = torch.tensor([[1, 2, 1], [4, 9, 3]])

# print(type(x[0, 2].item()))

# x = {1: "hello", 2: ["again", "hello"]}

# for key, value in enumerate(x):
#     print(f"Key : {key} | value : {value}")

test_path = Path().cwd()
print("Root dir : ", test_path)