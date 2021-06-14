---
title: Transformers for Image Recognition at Scale (Vision Transformers)
layout: post
tag: Computer Vision
---
# Transformers for Image Recognition at Scale
Google Research, Brain Team

## Vision Transformer (ViT)
* **2D image is reshaped into a sequence of flattened 2D patches.**
* flattened patches are linearly projected to an embedding representation
* 1D learnable positional embeddings are added (as opposed to fixed positional encodings)
* standard Transformer receives as input a 1D sequence of token embeddings
* similar to BERT's `[class]` token, prepend a learnable embedding $$z_0^0=x_{\text{class}}$$ whose state at the output of the Transformer encoder $$z_L^0$$ serves as the image representation $$y$$.
* a MLP classification head is attached to $$z_L^0$$ (two layers with a GELU non-linearity)

**Hybrid Architecture**: Input sequence is formed from feature maps of a CNN with patch size of 1 "pixel".

## Convolutional Inductive Bias
**Transformers lack some of the inductive biases inherent to CNN** (translation equivariance and locality) and do not generalize well when trained on insufficient amounts of data (i.e. ImageNet only).

Current state-of-the-art is held by *Noisy Student* (Xie et al., 2020) for ImageNet and *Big Transfer* (BiT) (Kolesnikov et al.) for other reported datasets (VITAB, CIFAR, ...)

ViT model pre-trained on larger datasets (14M-300M images, e.g. JFT-300M) outperforms ResNet (BiT) (Kolesnikov et al.) (by a percentage point or less).

When pre-trained on ImageNet, larger architectures underperform smaller architectures despite heavy regularization. Only with JFT-300M do we see the full benefit of larger models.

## Attention distance
* for each pixel, compute distance with other pixels weighted by the attention weight
* analogous to receptive field size in CNN (= filter size)
* some heads attend to most of the image already in the lowest layers, showing ability to integrate information globally
* other attention heads have small attention distances in the low layers (localized)
* highly localized attention is less pronounced in hybrid models suggesting that it may serve a similar function as early convolutional layers in CNNs

## Computational resources
Benefits from the efficient implementations on hardware accelerators (self-attention can be parallelized).

ViT uses 2-4x less compute to attain same performance as ResNets.

## Self-supervision
* masked patch prediction like BERT with masked language modeling
* predict 3-bit, mean color (i.e. out of 512 colors in total) of corrupted patches
* improves on training from scratch but still behind supervised pre-training (on ImageNet classification)

## Few-shot learning accuracy
Used for fast on-the-fly evaluation where fine-tuning would be too costly.

* a subset of training images are labeled with $$\{-1, 1\}^K$$ target vectors
* representations of those images are frozen
* goal is to solve a regularized linear regression model that maps the representations to the target vectors
* I assume, the more discriminative the representations are, the better the accuracy of the learned linear model.

## Follow-up:
learned positional embeddings (as introduced in BERT) vs positional encodings?
