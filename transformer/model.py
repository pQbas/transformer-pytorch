import torch
import torch.nn as nn
import math
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_length=5000):
        super().__init__()
        
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        
        pe = torch.zeros(max_seq_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, hidden_size]
        """
        return x + self.pe[:x.size(1)]


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.proj = nn.Linear(hidden_size, hidden_size) 
        self.dropout = nn.Dropout(dropout) 
        self.sqrtn = math.sqrt(self.head_dim)
        self.softmax = nn.Softmax(dim=-1)

    def out_operation(self, x):
        return self.proj(self.dropout(rearrange(x, 'b h n d -> b n (h d)')))

    def forward(self, q, k, v): 
        q  = rearrange(self.query(q) , 'b n (h d) -> b h n d', h = self.num_heads)
        v  = rearrange(self.value(v) , 'b n (h d) -> b h n d', h = self.num_heads)
        k_t = rearrange(self.key(k)  , 'b n (h d) -> b h d n', h = self.num_heads)
        z = self.softmax( (q @ k_t) / self.sqrtn ) @ v
        return self.out_operation(z)


class MaskedMultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(MaskedMultiHeadAttentionBlock, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.proj = nn.Linear(hidden_size, hidden_size) 
        self.dropout = nn.Dropout(dropout) 
        self.sqrtn = math.sqrt(self.head_dim)
        self.softmax = nn.Softmax(dim=-1)

    def out_operation(self, x): 
        return self.proj(self.dropout(rearrange(x, 'b h n d -> b n (h d)')))

    def apply_mask(self ,x ,mask):
        return x.masked_fill(mask == 0, float('-inf'))

    def forward(self, q, k, v, mask = None):

        if mask is None : 
            raise ValueError('Mask must be different than None')

        mask = mask.to(dtype=q.dtype, device=q.device) 
         
        q  = rearrange(self.query(q) , 'b n (h d) -> b h n d', h = self.num_heads)
        v  = rearrange(self.value(v) , 'b n (h d) -> b h n d', h = self.num_heads)
        k_t = rearrange(self.key(k)   , 'b n (h d) -> b h d n', h = self.num_heads)

        z_mask = self.softmax(self.apply_mask(( q @ k_t ) / self.sqrtn, mask)) @ v

        return self.out_operation(z_mask)


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size, feedforward_size, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size)
        )

    def forward(self, x):
        return self.feedforward(x)


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, attention_size, feedforward_size, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttentionBlock(hidden_size)
        self.feed_forward = FeedForwardBlock(hidden_size, feedforward_size, dropout)  
        self.norm1 = nn.LayerNorm(attention_size)
        self.norm2 = nn.LayerNorm(hidden_size)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm1(self.attention(x,x,x) + x)
        z  = self.norm2(self.feed_forward(x1) + x1)
        return z


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, attention_size, feedforward_size, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderBlock(hidden_size, attention_size, feedforward_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout=0.1):
        super(DecoderBlock, self).__init__() 
        self.masked_attention = MaskedMultiHeadAttentionBlock(hidden_size, num_heads, dropout)        
        self.enc_dec_attention = MultiHeadAttentionBlock(hidden_size, num_heads, dropout) 
        self.feed_forward = FeedForwardBlock(hidden_size, feedforward_size, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x, enc_output, tgt_mask=None):
        x1 = self.norm1(self.masked_attention(x, x, x, tgt_mask) + x)  
        x2 = self.norm2(self.enc_dec_attention(x1, enc_output, enc_output) + x1)
        z = self.norm3(self.feed_forward(x2) + x2)
        return z


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, feedforward_size, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, feedforward_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        sequence_length = 10,
        hidden_size = 64,
        attention_size = 64,
        embedding_size = 64,
        feedforward_size = 256,
        number_heads = 8,
        number_layers = 6,
        dropout = 0.1,
        vocab_size = 5000
        ):

        """Initialize a Transformer model for sequence-to-sequence tasks.

        This implements the standard Transformer architecture from "Attention is All You Need"
        (Vaswani et al., 2017) with both encoder and decoder stacks.

        Args:
        - sequence_length
            (int, optional) 
            Maximum length of input/output sequences. Defaults to 10.

        - hidden_size 
            (int, optional) 
            Dimension of model's hidden representations, also called embeding size. 
            Defaults to 64.

        - attention_size
            (int, optional) 
            Dimension of attention layers. Defaults to 64.

        - feedforward_size
            (int, optional), Defaults to 256 
            Dimension size of embedings inside FeedForwardBlock.

        - number_heads
            (int, optional)
            Number of attention heads in multi-head attention. Defaults to 8.

        - num_layers
            (int, optional)
            Number of encoder and decoder layers. Defaults to 6.

        - dropout (float, optional):
            Dropout rate for regularization. Defaults to 0.1.

        - vocab_size:
            (int, optional)
            Size of the model's vocabulary. Defaults to 5000.

        The model architecture includes:
            - Token embedding layer
            - Positional encoding
            - Multi-layer encoder with self-attention
            - Multi-layer decoder with masked self-attention and encoder-decoder attention
            - Output linear layer projecting to vocabulary size
        """

        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.embedding_size = embedding_size
        self.feedforward_size = feedforward_size
        self.number_heads = number_heads
        self.number_layers = number_layers
        self.dropout = dropout

        self.encoder = Encoder(
                num_layers=number_layers,
                hidden_size=hidden_size,
                attention_size=hidden_size, #attention_size,
                feedforward_size=feedforward_size,
                dropout=dropout
        ) 
        self.decoder = Decoder(
                num_layers = number_layers, 
                hidden_size = hidden_size, 
                num_heads = number_heads, 
                feedforward_size = feedforward_size,
                dropout = dropout
        )
        self.mask = torch.tril(torch.ones(sequence_length, sequence_length)).unsqueeze(0).unsqueeze(0)
        self.embeding_layer = nn.Linear(1, hidden_size)
        self.position_encoder = PositionalEncoding(hidden_size, sequence_length)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def embed_tokens(self, source, target):
        """ Embeds input sequences and adds positional encoding

        This function performs two main operations:
        1. Converts input tokens sequences (source and target) to embeddings using 
           a linear layer (self.embeding_layer)
        2. Adds positional encoding to help the model understand sequence order 
           (with self.positional_encoding) 

        Args:
        - source: 
            Tensor, shape [batch_size, seq_len, 1]
            Input sequence to be encoded (encoder input)

        - target: 
            Tensor, shape [batch_size, seq_len, 1]
            Target sequence to be decoded (decoder input)

        Returns:
        - tuple: A pair of tensors (source_embeding, target_embeding), where:

        - source_embeding: 
            Tensor, shape [batch_size, seq_len, hidden_size]
            Embedded source sequence with positional encoding

        - target_embeding:
            Tensor, shape [batch_size, seq_len, hidden_size]
            Embedded target sequence with positional encoding
                    
        """
        source_embeding = self.position_encoder(self.embeding_layer(source))
        target_embeding = self.position_encoder(self.embeding_layer(target))
        return source_embeding, target_embeding


    def forward(self, source, target):
        """Forward pass of the Transformer model.

        This function implements the complete Transformer architecture pipeline:
        1. Embedding Layer: Converts input token sequences into continuous vector 
           representations and adds positional encoding
        2. Encoder: Processes the source sequence through multiple layers of 
           self-attention and feed-forward networks to create a contextual 
           representation (z tensor)
        3. Decoder: Processes the target sequence using both self-attention 
           (with masking to prevent looking ahead) and cross-attention to the 
           encoder output
        4. Output Layer: Projects the decoder output to vocabulary size logits


        Args:
        - source: 
            Tensor, shape [batch_size, seq_len, 1]
            Input sequence to be translated/transformed

        - target:
            Tensor, shape [batch_size, seq_len, 1]
            Target sequence used for teacher forcing during training

        Returns:
        - logits: 
            Tensor, shape [batch_size, seq_len, vocab_size]
            Raw (non-normalized) probabilities for each token in vocabulary
            at each position in the sequence

        Note:
            The masking in the decoder ensures causality - each position can
            only attend to previous positions in the target sequence during
            self-attention, which is crucial for autoregressive generation.
        """
        source_embeding, target_embeding = self.embed_tokens(source, target)
        z = self.encoder(source_embeding)
        y = self.decoder(target_embeding, z, self.mask)
        logits = self.linear(y)
        return logits


if __name__ == '__main__':

    batch_size = 7
    seq_len = 10
    hidden_size = 64

    model = Transformer()
    x = torch.randn(batch_size, seq_len, 1)
    y = torch.randn(batch_size, seq_len, 1)

    probs = model(x, y)

    print(probs.shape)

