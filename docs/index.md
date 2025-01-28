# Documentation 

This is the documentation :V

Initialize a Transformer model for sequence-to-sequence tasks.

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
