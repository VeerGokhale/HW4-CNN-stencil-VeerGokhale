from __future__ import absolute_import
from preprocess import get_data, get_next_batch
from convolution.manual_convolution import ManualConv2d
from models.base_model import CifarModel

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


class CNN(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)
        self.hidden_layer_size = 128
        self.batch_size = 64

        # Conv1
        self.conv1 = nn.Conv2d(
            in_channels=self.image_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Conv2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # shared layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fc
        self.fc1 = nn.Linear(32 * 8 * 8, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.num_classes)

    def forward(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 3, 32, 32)
        :return: logits, a matrix of shape (num_inputs, num_classes)
        """

        x = self.conv1(inputs)      #(N, 3, 32, 32) --> (N, 16, 32, 32)
        x = self.relu(x)
        x = self.pool(x)            #(N, 16, 16, 16)

        x = self.conv2(x)           #(N, 32, 16, 16)
        x = self.relu(x)
        x = self.pool(x)            #(N, 32, 8, 8)

        x = torch.flatten(x, start_dim=1)   #(N, 32*8*8)
        x = self.fc1(x)             #(N, 128)
        x = self.relu(x)
        x = self.fc2(x)             #(N, num_classes)

        return x
