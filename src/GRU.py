import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Config(nn.Module):
    def __init__(
        self,
        vocab_size = 43,
        n_positions = 150,
        gru_layer = [256,512,1024],
        hidden_layer = 256,
        dropout = 0,
        layer_norm_epsilon = 1e-5,
        embedding_dim = 128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.gru_layer = gru_layer
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.embedding_dim = embedding_dim


class Encoder(nn.Module):
    def __init__(self,config):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        hidden_dims: list of int, the size of GRU hidden units
        output_dim: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.hidden_dims = config.gru_layer
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.output_dim = config.hidden_layer
        dims = self.hidden_dims.copy()
        dims.insert(0,self.embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx=0)
        self.gru = nn.ModuleList([nn.GRU(dims[i],dims[i+1],1) for i in range(len(self.hidden_dims))])
        self.linear = nn.Linear(sum(self.hidden_dims),self.output_dim)
        self.dropout = nn.Dropout(p=config.dropout)
    
    def forward(self,x,inference=False):
        # x: Tensor, [L,B]
        embedding = self.dropout(self.embedding(x)) # [L,B,E]
        states = []
        for v in self.gru:
            embedding, s = v(embedding) # [L,B,H]
            states.append(s.squeeze(0)) # [B,H]
        states = torch.cat(states,axis=1) # [B,Hsum]
        states = self.linear(states) # [B, Hout]
        if inference == False:
            states = states + torch.normal(0,0.05,size=states.shape).to(DEVICE)
        return embedding, torch.tanh(states)


class Decoder(nn.Module):
    def __init__(self,config):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        hidden_dims: list of int, the size of GRU hidden units
        bottleneck_dim: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.hidden_dims = config.gru_layer
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.bottleneck_dim = config.hidden_layer
        dims = self.hidden_dims.copy()
        dims.insert(0,self.embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx=0)
        self.linear = nn.Linear(self.bottleneck_dim,sum(self.hidden_dims))
        self.gru = nn.ModuleList([nn.GRU(dims[i],dims[i+1],1) for i in range(len(self.hidden_dims))])
        self.linear_out = nn.Linear(self.hidden_dims[-1],self.vocab_size,bias=False)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self,x,state,inference=False):
        # x: [L,B]
        # state: [B,H]
        embedding = self.dropout(self.embedding(x)) # [L,B,E]
        if inference == False:
            state = self.linear(state)
        state = state.unsqueeze(0) # [1,B,Hsum]
        states = []
        cur_state = 0
        for v,w in zip(self.gru,self.hidden_dims):
            embedding, s = v(embedding,state[:,:,cur_state:cur_state+w].contiguous())
            cur_state += w
            states.append(s.squeeze(0))
        output = self.linear_out(embedding)
        return output, torch.cat(states,axis=1)


class Seq2Seq(nn.Module):
    def __init__(self,config):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        hidden_dims: list of int, the size of GRU hidden units
        bottleneck_dim: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self,x,y,inference=False):
        _, latent = self.encoder(x,inference=inference)
        output, _ = self.decoder(y,latent)
        return output, latent

    def encode(self,x,inference=True):
        return self.encoder(x,inference=inference)[1]

    def decode(self,x,state,inference=False):
        return self.decoder(x,state,inference)