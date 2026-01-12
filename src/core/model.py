import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    Converts token IDs to dense vector embeddings.
    Scales embeddings by sqrt(d_model) as per the original Transformer paper.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings using sine and cosine functions.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix [seq_len, d_model]
        pe = torch.zeros(seq_len, d_model)
        
        # Position indices [seq_len, 1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Scaling factors for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, seq_len, d_model]
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Layer normalization for stabilizing training.
    """
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                  mask: torch.Tensor = None, dropout: nn.Module = None):
        d_k = query.shape[-1]
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attn_weights = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        return self.w_o(x)

class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization (Pre-LN).
    """
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    Single Transformer encoder block.
    """
    def __init__(self, d_model: int, self_attn: MultiHeadAttention, 
                 feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
    
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.residual2(x, self.feed_forward)
        return x

class Encoder(nn.Module):
    """
    Transformer encoder with multiple layers.
    """
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    Single Transformer decoder block.
    """
    def __init__(self, d_model: int, self_attn: MultiHeadAttention, 
                 cross_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        x = self.residual3(x, self.feed_forward)
        return x

class Decoder(nn.Module):
    """
    Transformer decoder with multiple layers.
    """
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Projects decoder output to vocabulary size.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class Transformer(nn.Module):
    """
    Complete Transformer model.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, d_ff: int = 2048, 
                      num_heads: int = 8, 
                      num_encoder_layers: int = 6,
                      num_decoder_layers: int = 6,
                      dropout: float = 0.1) -> Transformer:
    """
    Build a complete Transformer model with specified parameters.
    """
    # Create embeddings
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    
    # Create positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create encoder blocks
    encoder_blocks = []
    for _ in range(num_encoder_layers):
        self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        ff = FeedForward(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(d_model, self_attn, ff, dropout))
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(num_decoder_layers):
        self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        ff = FeedForward(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(d_model, self_attn, cross_attn, ff, dropout))
    
    # Create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create projection layer
    projection = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, 
                             src_pos, tgt_pos, projection)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer