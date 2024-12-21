import torch
import torch.nn as nn
import math
from einops import rearrange

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


    def forward(self, q, k, v):

        sqrtn  = self.sqrtn
        sftmx  = lambda x : self.softmax(x)
        outop  = lambda x : self.proj(self.dropout(rearrange(x, 'b h n d -> b n (h d)')))
        
        Q  = rearrange(self.query(q) , 'b n (h d) -> b h n d', h = self.num_heads)
        V  = rearrange(self.value(v) , 'b n (h d) -> b h n d', h = self.num_heads)
        KT = rearrange(self.key(k)   , 'b n (h d) -> b h d n', h = self.num_heads)

        return outop( sftmx((Q @ KT) / sqrtn) @ V )


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


    def forward(self, q, k, v, mask = None):

        if mask == None : raise ValueError('Mask must be different than None')

        mask = mask.to(dtype=q.dtype, device=q.device)

        sqrtn  = self.sqrtn
        sftmx  = lambda x : self.softmax(x)
        outop  = lambda x : self.proj(self.dropout(rearrange(x, 'b h n d -> b n (h d)')))
        maskf  = lambda x, mask : x.masked_fill(mask == 0, float('-inf'))
          
        Q  = rearrange(self.query(q) , 'b n (h d) -> b h n d', h = self.num_heads)
        V  = rearrange(self.value(v) , 'b n (h d) -> b h n d', h = self.num_heads)
        KT = rearrange(self.key(k)   , 'b n (h d) -> b h d n', h = self.num_heads)

        return outop(sftmx(maskf(Q @ KT / sqrtn, mask)) @ V)


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        MLP1 = lambda x : self.linear1(x)
        MLP2 = lambda x : self.linear2(x)
        ReLU = lambda x : self.relu(x)
        drop = lambda x : self.dropout(x)

        return MLP2(drop(ReLU(MLP1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, attention_size, ff_size, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttentionBlock(hidden_size)
        self.ff_block = FeedForwardBlock(hidden_size, ff_size, dropout)
        
        self.norm1 = nn.LayerNorm(attention_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        drop  = lambda t : self.dropout(t)
        norm1 = lambda t : self.norm1(t)
        norm2 = lambda t : self.norm2(t)

        MultiheadAttention = lambda t : drop(self.attention(t, t, t))
        FeedForward = lambda t : drop(self.ff_block(t))

        x1 = norm1(MultiheadAttention(x) + x)
        z  = norm2(FeedForward(x1) + x1)
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
        self.ff_block = FeedForwardBlock(hidden_size, ff_size, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x, enc_output, tgt_mask=None):
        x1 = self.norm1(self.masked_attention(x, x, x, tgt_mask) + x)  
        x2 = self.norm2(self.enc_dec_attention(x1, enc_output, enc_output) + x1)
        z = self.norm3(self.ff_block(x2) + x2)
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


if __name__ == '__main__':

    batch_size = 2
    seq_len = 10
    hidden_size = 64
    num_heads = 8
    ff_size = 256
    num_layers = 6
    attention_size = 64
    dropout = 0.1

    encoder = Encoder(
        num_layers=num_layers,
        hidden_size=hidden_size,
        attention_size=attention_size,
        ff_size=ff_size,
        dropout=dropout,
    )

    decoder = Decoder(
        num_layers = num_layers, 
        hidden_size = hidden_size, 
        num_heads = num_heads, 
        ff_size = ff_size,
        dropout = dropout
    )

    x = torch.randn(batch_size, seq_len, hidden_size)
    y = torch.randn(batch_size, seq_len, hidden_size)

    z = encoder(x)

    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    probs = decoder(y, z, mask)

    print(probs.shape)

