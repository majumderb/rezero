# ReZero for Deep Neural Networks

[**ReZero is All You Need: Fast Convergence at Large Depth**](https://arxiv.org/abs/2003.04887).\*Uncertainty in AI (UAI), 2021*.\Thomas Bachlechner*, Bodhisattwa Prasad Majumder*, Huanru Henry Mao*, Garrison W. Cottrell, Julian McAuley (* denotes equal contributions)

This repository contains the ReZero-Transformer implementation from the paper. It matches Pytorch's Transformer and can be easily used as a drop-in replacement. 

Quick Links:
- [Installation](https://github.com/majumderb/rezero/blob/master/README.md#installation)
- [Usage](https://github.com/majumderb/rezero/blob/master/README.md#usage)
- [Tutorials](https://github.com/majumderb/rezero/blob/master/README.md#tutorials)
- [Citation](https://github.com/majumderb/rezero/blob/master/README.md#citation)

# Abstract

Deep networks have enabled significant performance gains across domains, but they often suffer from vanishing/exploding gradients. This is especially true for Transformer architectures where depth beyond 12 layers is difficult to train without large datasets and computational budgets. In general, we find that inefficient signal propagation impedes learning in deep networks. In Transformers, multi-head self-attention is the main cause of this poor signal propagation. To facilitate deep signal propagation, we propose **ReZero**, a simple change to the architecture that initializes an arbitrary layer as the identity map, using a single additional learned parameter per layer. We apply this technique to language modeling and find that we can easily train ReZero-Transformer networks over a hundred layers. When applied to 12 layer Transformers, ReZero converges 56% faster on enwiki8. ReZero applies beyond Transformers to other residual networks, enabling 1,500% faster convergence for deep fully connected networks and 32% faster convergence for a ResNet-56 trained on CIFAR 10.

# Installation
Simply install from pip:

```
pip install rezero
```

Pytorch 1.4 or greater is required.

# Usage
We provide custom ReZero Transformer layers (RZTX).

For example, this will create a Transformer encoder:
```py
import torch
import torch.nn as nn
from rezero.transformer import RZTXEncoderLayer

encoder_layer = RZTXEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
```

This will create a Transformer decoder:
```py
import torch
import torch.nn as nn
from rezero.transformer import RZTXDecoderLayer

decoder_layer = RZTXDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
```

Make sure `norm` argument is left as `None` as to not use `LayerNorm` in the Transformer.

See https://pytorch.org/docs/master/nn.html#torch.nn.Transformer for details on how to integrate customer Transformer layers to Pytorch.

# Tutorials

1. [Training 128 layer ReZero Transformer on WikiText-2 language modeling](https://github.com/tbachlechner/ReZero-examples/blob/master/ReZero-Deep_Fast_Transformer.ipynb)
2. [Training 10,000 layer ReZero neural network on CIFAR-10 data](https://github.com/tbachlechner/ReZero-examples/blob/master/ReZero-Deep_Fast_NeuralNetwork.ipynb)

Watch for more tutorials in this [space](https://github.com/tbachlechner/ReZero-examples).

# Citation
If you find `rezero` useful for your research, please cite our paper:
```BibTex
@inproceedings{BacMajMaoCotMcA20,
    title = "ReZero is All You Need: Fast Convergence at Large Depth",
    author = "Bachlechner, Thomas  and
      Majumder, Bodhisattwa Prasad
      Mao, Huanru Henry and
      Cottrell, Garrison W. and
      McAuley, Julian",
    booktitle = "arXiv",
    year = "2020",
    url = "https://arxiv.org/abs/2003.04887"
}
```
