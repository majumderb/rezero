# ReZero for Deep Neural Networks

ReZero for Speed and Depth: Delving Deep into Transformers; *ArXiv, March 2020*.

Thomas Bachlechner*, Bodhisattwa Prasad Majumder*, Huanru Henry Mao*, Garrison W. Cottrell, Julian McAuley (* denotes equal contributions)

# Abstract

We propose **ReZero**, a simple change to the residual connection that initializes the residual function to zero using one learned scalar multiplier per layer. We apply this technique to character-level language modeling and find that we can easily train ReZero-Transformer networks up to at least 200 layers. When applied to a 12 layer Transformers, ReZero converges 83% faster on enwiki8. ReZero applies beyond Transformers to general residual networks. For example, we find that ReZero enables  32% faster convergence for a ResNet-56 trained on CIFAR 10.

# Code coming soon! 
