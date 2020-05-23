#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab, debug=False):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.char_embed_size = 50
        self.word_embed_size = word_embed_size
        self.embedding = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=vocab.char2id['<pad>'])
        self.conv_layer = CNN(self.char_embed_size, word_embed_size)
        self.hwy_layer = Highway(word_embed_size)
        self.dropout = nn.Dropout(0.3)
        self._debug = debug

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        if self._debug:
            print("sentence_length: {}\nbatch_size: {}\nmax_word_length: {}\nchar_embed_size: {}\nembed_size: {}"
                  .format(input.shape[0], input.shape[1], input.shape[2], self.char_embed_size, self.word_embed_size))
            print("input.shape: " + str(input.shape))

        sentence_len, batch_size, max_word_len = input.shape
        x_char_embeddings = self.embedding(input)  # (sentence_length, batch_size, max_word_length, char_embed_size)

        x_reshaped = x_char_embeddings.view(sentence_len*batch_size, max_word_len, self.char_embed_size).squeeze(2)\
            .permute(0, 2, 1)
        x_conv_out = self.conv_layer(x_reshaped)
        x_highway = self.hwy_layer(x_conv_out)
        x_dropout = self.dropout(x_highway)
        x_out = x_dropout.view(sentence_len, batch_size, self.word_embed_size)

        # if self._debug:
        #     print("x_char_embeddings.shape: " + str(x_char_embeddings.shape))
        #     print("x_.shape: " + str(x_.shape))
        #     print("x_reshaped.shape: " + str(x_reshaped.shape))
        #     print("x_conv_out.shape: " + str(x_conv_out.shape))
        #     print("x_highway.shape: " + str(x_highway.shape))
        #     print("x_dropout.shape: " + str(x_dropout.shape))
        #     print("x_out.shape: " + str(x_out.shape))

        return x_out

        ### END YOUR CODE

