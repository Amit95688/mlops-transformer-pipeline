import torch 
import torch.nn as nn
import math

class input_embedding(nn.Module):
    """
    Converts token IDs to dense vector embeddings.
    Scales embeddings by sqrt(d_model) as per the original Transformer paper.
    """
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model          # d_model: embedding dimension (e.g., 512)
        self.vocab_size = vocab_size    # vocab_size: number of unique tokens in vocabulary (e.g., 10000)
        # embedding: lookup table of shape [vocab_size, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: input token IDs, shape [batch_size, seq_len]
        # returns: embedded vectors, shape [batch_size, seq_len, d_model]
        # multiply by sqrt(d_model) to scale embeddings (from original paper)
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    
class positional_encoding(nn.Module):
    """
    Adds positional information to embeddings using sine and cosine functions.
    Allows the model to understand token positions in the sequence.
    """
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model      # d_model: embedding dimension (must match embedding layer)
        self.seq_len = seq_len      # seq_len: maximum sequence length (e.g., 512)
        self.dropout = nn.Dropout(dropout)  # dropout: regularization rate (e.g., 0.1)

        # pe: positional encoding matrix, shape [d_model, seq_len]
        pe = torch.zeros(d_model, seq_len)
        
        # position: vector of positions [0, 1, 2, ..., seq_len-1], shape [seq_len, 1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # div_term: scaling factors for different dimensions, shape [d_model/2]
        # formula: 1 / (10000^(2i/d_model)) where i is the dimension index
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # apply sine to even indices (0, 2, 4, ...)
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[0::2, :] = torch.sin(position * div_term).transpose(0, 1)
        
        # apply cosine to odd indices (1, 3, 5, ...)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[1::2, :] = torch.cos(position * div_term).transpose(0, 1)
        
        # add batch dimension: shape [1, d_model, seq_len]
        pe = pe.unsqueeze(0)
        
        # register_buffer: saves pe as part of model state but not as a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: input embeddings, shape [batch_size, seq_len, d_model]
        # add positional encodings to input (no gradients for positional encodings)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        # returns: embeddings with positional info + dropout applied
        return self.dropout(x)

class layerNormalization(nn.Module):
    """
    Normalizes inputs across the feature dimension to stabilize training.
    Formula: output = alpha * (x - mean) / (std + eps) + beta
    """
    def __init__(self, eps:float=1e-6):
        super().__init__()
        self.eps = eps  # eps: small value to prevent division by zero (e.g., 1e-6)
        # alpha: learnable scale parameter (initialized to 1)
        self.alpha = nn.Parameter(torch.ones(1))
        # beta: learnable shift parameter (initialized to 0)
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: input tensor, shape [batch_size, seq_len, d_model]
        # mean: average across the last dimension (d_model), shape [batch_size, seq_len, 1]
        mean = x.mean(-1, keepdim=True)
        # std: standard deviation across the last dimension, shape [batch_size, seq_len, 1]
        std = x.std(-1, keepdim=True)
        # normalize, scale, and shift: output shape [batch_size, seq_len, d_model]
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

class feedForward(nn.Module):
    """
    Position-wise feed-forward network (FFN) applied after attention.
    Two linear transformations with ReLU activation: FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        # linear1: first linear layer expands from d_model to d_ff (e.g., 512 -> 2048)
        self.linear1 = nn.Linear(d_model, d_ff)
        # dropout: regularization applied between the two linear layers
        self.dropout = nn.Dropout(dropout)
        # linear2: second linear layer projects back from d_ff to d_model (e.g., 2048 -> 512)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: input tensor, shape [batch_size, seq_len, d_model]
        # expand: shape [batch_size, seq_len, d_ff]
        x = self.linear1(x)
        # activation: ReLU(x) = max(0, x)
        x = torch.relu(x)
        # regularization: randomly zero out some neurons
        x = self.dropout(x)
        # project back: shape [batch_size, seq_len, d_model]
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism allowing the model to focus on different parts of the input.
    Splits input into multiple heads, applies scaled dot-product attention, and concatenates results.
    """
    def __init__(self, d_model:int, num_heads:int, dropout:float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model          # d_model: embedding dimension (e.g., 512)
        self.num_heads = num_heads      # num_heads: number of attention heads (e.g., 8)
        self.d_k = d_model // num_heads # d_k: dimension per head (e.g., 64)
        
        # linear layers for query, key, value projections
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        # final linear layer after concatenation of heads
        self.linear_out = nn.Linear(d_model, d_model)
        
        # dropout for attention weights
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, d_k:int, mask:torch.Tensor=None, dropout:nn.Module=None)->torch.Tensor:
        # scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask ==0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        
        if dropout is not None:
            attn = dropout(attn)
        
        output = torch.matmul(attn, value)  # [batch_size, num_heads, seq_len, d_k]
        return output, attn
    
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        batch_size = query.size(0)
        seq_len = query.size(1)  # seq_len

        # project inputs to multi-heads
        Q = self.linear_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = self.linear_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)    # [batch_size, num_heads, seq_len, d_k]
        V = self.linear_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        
        x,self_attn_score=MultiHeadAttention.attention(Q, K, V, self.d_k, mask, self.dropout)
        x=x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)
        # concatenate heads and project back
        x = self.linear_out(x)
        return x    
    
class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization.
    Helps in training deep networks by allowing gradients to flow through the network.
    """
    def __init__(self, d_model:int, dropout:float):
        super().__init__()
        self.layer_norm = layerNormalization()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor, sublayer:nn.Module)->torch.Tensor:
        # apply layer normalization
        normalized_x = self.layer_norm(x)
        # apply the sublayer (e.g., attention or feed-forward)
        sublayer_output = sublayer(normalized_x)
        # apply dropout
        dropped_output = self.dropout(sublayer_output)
        # add residual connection
        return x + dropped_output
    
class Encoderlayer(nn.Module):
    """
    Transformer encoder block.
    Consists of multi-head self-attention and position-wise feed-forward network, each followed by residual connections.
    """
    def __init__(self,self_attn:MultiHeadAttention, feed_forward:feedForward, dropout:float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(self_attn.d_model, dropout)
        self.residual2 = ResidualConnection(self_attn.d_model, dropout)
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        # apply multi-head self-attention with residual connection
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask))
        # apply position-wise feed-forward network with residual connection
        x = self.residual2(x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder blocks.
    Stacks several encoder blocks to build a deep encoder.
    """
    def __init__(self, encoder_block:Encoderlayer, num_layers:int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_block for _ in range(num_layers)])
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        # pass input through each encoder block sequentially
        for layer in self.layers:
            x = layer(x, mask)
        return x.norm()  # final layer normalization

class EncoderLayer(nn.Module):
    """
    Complete Transformer encoder model.
    Combines input embedding, positional encoding, and stacked encoder blocks.
    """
    def __init__(self, self_attn:MultiHeadAttention, feed_forward:feedForward,  dropout:float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.resdual_coonections=nn.Module([ResidualConnection(self_attn.d_model, dropout) for _ in range(2)])

    def forward(self, x:torch.Tensor, encoder_output:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        # apply multi-head self-attention with residual connection
        x = self.resdual_coonections[0](x, lambda x: self.self_attn(x, x, x, mask))
        # apply position-wise feed-forward network with residual connection
        x = self.resdual_coonections[1](x, self.feed_forward)
        return x