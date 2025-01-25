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
        k_t = rearrange(self.key(k)   , 'b n (h d) -> b h d n', h = self.num_heads)
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
    def __init__(self, hidden_size, ff_size, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size)
        )

    def forward(self, x):
        return self.feedforward(x)


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, attention_size, ff_size, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttentionBlock(hidden_size)
        self.feed_forward = FeedForwardBlock(hidden_size, ff_size, dropout)  
        self.norm1 = nn.LayerNorm(attention_size)
        self.norm2 = nn.LayerNorm(hidden_size)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm1(self.attention(x,x,x) + x)
        z  = self.norm2(self.feed_forward(x1) + x1)
        return z


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, attention_size, ff_size, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderBlock(hidden_size, attention_size, ff_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout=0.1):
        super(DecoderBlock, self).__init__() 
        self.masked_attention = MaskedMultiHeadAttentionBlock(hidden_size, num_heads, dropout)        
        self.enc_dec_attention = MultiHeadAttentionBlock(hidden_size, num_heads, dropout) 
        self.feed_forward = FeedForwardBlock(hidden_size, ff_size, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x, enc_output, tgt_mask=None):
        x1 = self.norm1(self.masked_attention(x, x, x, tgt_mask) + x)  
        x2 = self.norm2(self.enc_dec_attention(x1, enc_output, enc_output) + x1)
        z = self.norm3(self.feed_forward(x2) + x2)
        return z


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, ff_size, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, ff_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len = 10,
        hidden_size = 64,
        attention_size = 64,
        num_heads = 8,
        ff_size = 256,
        num_layers = 6,
        dropout = 0.1,
        vocab_size = 5000
        ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
                num_layers=num_layers,
                hidden_size=hidden_size,
                attention_size=attention_size,
                ff_size=ff_size,
                dropout=dropout
        ) 
        self.decoder = Decoder(
                num_layers = num_layers, 
                hidden_size = hidden_size, 
                num_heads = num_heads, 
                ff_size = ff_size,
                dropout = dropout
        )
        self.mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        self.embd = nn.Linear(1, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, seq_len)
        self.linear = nn.Linear(hidden_size, vocab_size)


    def embd_inputs(self, src, tgt):
        src_embd = self.pos_encoder(self.embd(src))
        tgt_embd = self.pos_encoder(self.embd(tgt))
        return src_embd, tgt_embd


    def forward(self, src, tgt):

        src_embd, tgt_embd = self.embd_inputs(src, tgt)
        z = self.encoder(src_embd)
        y = self.decoder(tgt_embd, z, self.mask)
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

