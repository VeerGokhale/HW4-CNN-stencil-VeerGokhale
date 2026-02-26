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
        self.input_width = ???
        self.input_height = ???
        self.image_channels = ???
        self.num_classes = len(classes)
        self.hidden_layer_size = 256 # feel free to change this ;)
        self.batch_size = 256 # feel free to change this ;)


        # TODO: Initialize your convolutional and linear layers here

    def forward(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 3, 32, 32)
        :return: logits, a matrix of shape (num_inputs, num_classes)
        """

        # TODO: Implement your forward pass here
        raise NotImplementedError("Implement me!")
