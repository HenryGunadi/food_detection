import torch.nn as nn
import torch
from typing import List

class Cnn_v1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: List[int], output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, # RGB channel shape (3)
                      out_channels=hidden_units[0],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units[0],
                      out_channels=hidden_units[0],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units[0],
                      out_channels=hidden_units[1],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units[1],
                      out_channels=hidden_units[1],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units[1],
                      out_channels=hidden_units[2],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units[2]),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units[2],
                      out_channels=hidden_units[2],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        # self.conv_block_4 = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_units[2],
        #               out_channels=hidden_units[3],
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(hidden_units[3]),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=hidden_units[3],
        #               out_channels=hidden_units[3],
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(hidden_units[3]),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,
        #                  stride=2)
        # )
        # self.conv_block_5 = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_units[3],
        #               out_channels=hidden_units[4],
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(hidden_units[4]),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=hidden_units[4],
        #               out_channels=hidden_units[4],
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(hidden_units[4]),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,
        #                  stride=2)
        # )
        # fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,
                      out_features=output_shape),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))