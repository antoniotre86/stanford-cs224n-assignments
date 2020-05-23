#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):

    def __init__(self, word_embedding_size: int):
        """

        Args:
            word_embedding_size:
        """
        super(Highway, self).__init__()
        self.proj_layer = nn.Linear(word_embedding_size, word_embedding_size, bias=True)
        self.gate_layer = nn.Linear(word_embedding_size, word_embedding_size, bias=True)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x_conv_out:

        Returns:

        """
        x_proj = F.relu(self.proj_layer(x_conv_out))
        x_gate = torch.sigmoid(self.gate_layer(x_conv_out))
        x_highway = x_gate * x_proj + (1-x_gate) * x_conv_out
        return x_highway


    ### END YOUR CODE

