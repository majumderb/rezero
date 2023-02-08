from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class RZTXEncoderLayer(nn.Module):
    r"""RZTXEncoderLayer is made up of self-attn and feedforward network with
    residual weights for faster convergece.
    This encoder layer is based on the paper "ReZero is All You Need:
    Fast Convergence at Large Depth".
    Thomas Bachlechner∗, Bodhisattwa Prasad Majumder∗, Huanru Henry Mao∗,
    Garrison W. Cottrell, Julian McAuley. 2020.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_res_init: Use residual initialization
    Examples::
        >>> encoder_layer = RZTXEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', batch_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.zeros((1,), **factory_kwargs))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (torch.Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in PyTroch Transformer class.
        """
        # Self attention layer
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0]  # no attention weights
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)

        # Pointiwse FF Layer
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src


class RZTXDecoderLayer(nn.Module):
    r"""RZTXDecoderLayer is made up of self-attn and feedforward network with
    residual weights for faster convergece.
    This encoder layer is based on the paper "ReZero is All You Need:
    Fast Convergence at Large Depth".
    Thomas Bachlechner∗, Bodhisattwa Prasad Majumder∗, Huanru Henry Mao∗,
    Garrison W. Cottrell, Julian McAuley. 2020.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_res_init: Use residual initialization
    Examples::
        >>> decoder_layer = RZTXDecoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = decoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', batch_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model,
                                               nhead,
                                               dropout=dropout,
                                               batch_first=batch_first,
                                               **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.zeros((1,), **factory_kwargs))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in PyTroch Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2) * self.resweight
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2) * self.resweight

        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2) * self.resweight
        return tgt


def Transformer(*, num_encoder_layers, num_decoder_layers, **kw):
    layer_keywords = {'d_model', 'nhead', 'dim_feedforward', 'dropout',
                      'activation', 'batch_first', 'device', 'dtype'}
    layer_kwargs = {k: v for k, v in kw.items() if k in layer_keywords}
    encoder_layer = RZTXEncoderLayer(**layer_kwargs)
    decoder_layer = RZTXDecoderLayer(**layer_kwargs)
    encoder_norm = None
    decoder_norm = None
    encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
    return nn.Transformer(**kw, custom_encoder=encoder, custom_decoder=decoder)
