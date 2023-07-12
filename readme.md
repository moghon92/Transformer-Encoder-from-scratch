# Single-Layer Transformer for Sequence Encoding and Binary Classification from scratch

This project presents my own implementation of a single-layer Transformer model encoder with multi-headed attention. The Transformer is designed to encode a sequence of text and perform binary classification tasks. It operates on minibatches of size N and has the following specifications: 

- Vocabulary size (V): The model works with a vocabulary of V unique words.
- Sequence length (T): It processes sequences of length T.
- Hidden dimension (H): The model has a hidden dimension of H.
- Word vectors dimension (H): The word vectors used in the model are also of dimension H.

## Contents

- [Overview](#overview)
- [Implementation Details](#implementation-details)
- [Model Architecture](#model-architecture)

## Overview

The single-layer Transformer is a powerful architecture that leverages self-attention mechanisms to capture contextual relationships within a sequence. It consists of an encoder that processes the input sequence and a binary classifier that predicts the class label. The Transformer allows for parallel computation and effectively captures long-range dependencies.

## Implementation Details

The implementation consists of the following files:

- `transformer.py`: Contains the implementation of the single-layer Transformer model, including the self-attention mechanism, positional encoding, feed-forward layers, and binary classifier.



## Model Architecture

The architecture of the single-layer Transformer for sequence encoding and binary classification is depicted in the following diagram:

![Single-Layer Transformer Architecture](https://i0.wp.com/kikaben.com/wp-content/uploads/2022/04/image-443.png?w=800&ssl=1)

