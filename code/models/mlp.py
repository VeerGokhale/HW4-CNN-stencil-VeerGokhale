from __future__ import absolute_import
from models.base_model import CifarModel

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


class MLP(CifarModel):
    def __init__(self, class_idxs):
        """
        This model class will contain the architecture for your MLP that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(MLP, self).__init__()

        # Initialize all hyperparameters
        self.input_width = ???
        self.input_height = ???
        self.image_channels = ???
        self.num_classes = len(class_idxs)
        self.hidden_layer_size = 128 # feel free to change this ;)
        self.batch_size = 64 # feel free to change this ;)

        # TODO: Initialize your layers here

    def forward(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 3, 32, 32)
        :return: logits, a matrix of shape (num_inputs, num_classes)
        """
        # TODO: Implement your forward pass here
        raise NotImplementedError("Implement me!")
