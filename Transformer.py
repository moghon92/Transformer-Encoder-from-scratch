import numpy as np

import torch
from torch import nn
import random


class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)

        self.token_emb = nn.Embedding(input_size, self.word_embedding_dim).to(device)
        self.pos_emb = nn.Embedding(max_length, self.word_embedding_dim).to(device)

        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k).to(device)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v).to(device)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q).to(device)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k).to(device)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v).to(device)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q).to(device)
        
        self.softmax = nn.Softmax(dim=2).to(device)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim).to(device)
        self.norm_mh = nn.LayerNorm(self.hidden_dim).to(device)

        

        self.ff_lin1 = nn.Linear(hidden_dim, dim_feedforward).to(device)
        self.activation = nn.ReLU().to(device)
        self.ff_lin2 = nn.Linear(dim_feedforward, hidden_dim).to(device)
        self.ff_norm = nn.LayerNorm(hidden_dim).to(device)

        self.out_linear = nn.Linear(hidden_dim, output_size).to(device)


        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

  
        outputs = self.embed(inputs)
        outputs = self.multi_head_attention(outputs)
        outputs = self.feedforward_layer(outputs)
        outputs = self.final_layer(outputs)

        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        embeddings = None

        tok_emb = self.token_emb(inputs)

        seq_len = inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.expand_as(inputs).to(self.device)
        pos_emb = self.pos_emb(pos)

        embeddings = tok_emb + pos_emb

        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """

        # attention 1
        k1 = self.k1(inputs)
        q1 = self.q1(inputs)
        v1 = self.v1(inputs)
        d_k1 = k1.size(-1)

        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / np.sqrt(d_k1)
        b_attn1 = self.softmax(scores1)
        atten1 = torch.matmul(b_attn1, v1)

        # attention 2
        k2 = self.k2(inputs)
        q2 = self.q2(inputs)
        v2 = self.v2(inputs)
        d_k2 = k2.size(-1)

        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / np.sqrt(d_k2)
        b_attn2 = self.softmax(scores2)
        atten2 = torch.matmul(b_attn2, v2)

        outputs = torch.cat((atten1, atten2), dim=-1)
        outputs = self.attention_head_projection(outputs)
        outputs = self.norm_mh(outputs+inputs)

        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        

        outputs = self.ff_lin1(inputs)
        outputs = self.activation(outputs)
        outputs = self.ff_lin2(outputs)
        outputs = self.ff_norm(outputs+inputs)

        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        outputs = self.out_linear(inputs)

        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True