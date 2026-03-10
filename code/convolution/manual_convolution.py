from __future__ import absolute_import
from preprocess import unpickle, get_next_batch, get_data

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


class ManualConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, use_bias: bool = True, trainable: bool = True):
        """
        Manual 2D Convolution layer because you are so cool.

        :param in_channels: number of input channels
        :param out_channels: number of output channels (filters)
        :param kernel_size: size of the convolving kernel (assumes square kernel)
        :param stride: stride of the convolution (default: 1)
        :param padding: zero-padding added to both sides of the input (default: 0)
        :param use_bias: whether to use bias (default: True)
        :param trainable: whether the weights are trainable (default: True)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.bias = None
        self.trainable = trainable

        # NOTE: DO NOT EDIT
        self._initalize_weights()

    def _initalize_weights(self):
        """
        NOTE: DO NOT EDIT!
        We initalize the weights for you here! Do not modify this code, but you should take a 
        moment to read it and understand what it does.
        """
        filters = torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        nn.init.trunc_normal_(filters, std=0.1) # we initialize the weights to be truncated normal

        if self.trainable:
            self.filters = nn.Parameter(filters)
        else:
            self.register_buffer('filters', filters) # buffer just means that it is not a parameter that will be updated during training

        if self.use_bias:
            bias = torch.empty(self.out_channels)
            nn.init.trunc_normal_(bias, std=0.1)
            if self.trainable:
                self.bias = nn.Parameter(bias)
            else:
                self.register_buffer('bias', bias)
            

    def get_weights(self):
        if self.bias is not None:
            return self.filters, self.bias
        return self.filters

    def set_weights(self, filters, bias=None):
        """Set the weights of the layer."""
        if isinstance(filters, nn.Parameter):
            filters = filters.data
        if isinstance(self.filters, nn.Parameter):
            self.filters.data = filters.clone()
        else:
            self.filters = filters.clone()

        if bias is not None:
            if isinstance(bias, nn.Parameter):
                bias = bias.data
            if isinstance(self.bias, nn.Parameter):
                self.bias.data = bias.clone()
            else:
                self.bias = bias.clone()

    def forward(self, inputs):
        """
        Perform manual 2D convolution.

        :param inputs: tensor with shape [num_examples, in_channels, in_height, in_width] 
        :return: output tensor with shape [num_examples, out_channels, out_height, out_width]
        """
        # Get dimensions
        num_examples, input_in_channels, in_height, in_width = inputs.shape
        filter_out_channels, filter_in_channels, filter_height, filter_width = self.filters.shape

        if input_in_channels != filter_in_channels:
            raise ValueError(f"Input channels is {input_in_channels} "
                           f"but filter expects {filter_in_channels}")

            # Pad input on height and width dimensions
        if self.padding > 0:
            padded_inputs = F.pad(inputs, (self.padding, self.padding, self.padding, self.padding))
        else:
            padded_inputs = inputs

        _, _, padded_height, padded_width = padded_inputs.shape

        out_height = (padded_height - filter_height) // self.stride + 1
        out_width = (padded_width - filter_width) // self.stride + 1

        outputs = torch.zeros(
            (num_examples, filter_out_channels, out_height, out_width),
            dtype=inputs.dtype,
            device=inputs.device
        )

        for n in range(num_examples):
            for oc in range(filter_out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + filter_height
                        w_start = j * self.stride
                        w_end = w_start + filter_width

                        patch = padded_inputs[n, :, h_start:h_end, w_start:w_end]
                        outputs[n, oc, i, j] = torch.sum(patch * self.filters[oc])

                        if self.use_bias:
                            outputs[n, oc, i, j] += self.bias[oc]

        return outputs
