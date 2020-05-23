#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    kernel_size = 5
    padding = 1

    def __init__(self, char_embedding_size: int, word_embedding_size: int):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=char_embedding_size,
            out_channels=word_embedding_size,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x_reshaped: torch.Tensor

        Returns:
            x_conv_out: torch.Tensor
        """

        x_conv = self.conv_layer(x_reshaped)
        x_conv_out = F.relu(x_conv).max(dim=2)[0]
        return x_conv_out



