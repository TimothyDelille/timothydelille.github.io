---
layout: post
title: Self-supervised learning - Yann LeCun (03/04/2021)
tag: Deep Learning
---
# Self-supervised learning: the dark matter of intelligence
Yann LeCun – 03/04/2021

## Overview
* Supervised learning is a bottleneck for building more intelligent generalist models that can do multiple tasks and acquire new skills without massive amounts of labeled data
* generalized knowledge about the world, or common sense, forms the bulk of biological intelligence in both humans and animals. In a way, common sense is the dark matter of artificial intelligence. (dark matter is implied by calculations showing that many galaxies would fly apart if they did not contain a large amount of unseen matter)
* Common sense helps people learn new skills without requiring massive amounts of teaching for every single task.

We believe that self-supervised learning (SSL) is one of the most promising ways to build such background knowledge and approximate a form of common sense in AI systems.

## History in NLP
The term “self-supervised learning” is more accepted than the previously used term “unsupervised learning.” Unsupervised learning is an ill-defined and misleading term that suggests that the learning uses no supervision at all. In fact, self-supervised learning is not unsupervised, as it uses far more feedback signals than standard supervised and reinforcement learning methods do.

Self-supervised learning has long had great success in advancing the field of natural language processing (NLP):
* [Collobert-Weston 2008 model](https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf)
* [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)
* [GloVE](https://nlp.stanford.edu/pubs/glove.pdf)
* [fastText](https://arxiv.org/pdf/1607.01759.pdf)
* [BERT]( https://arxiv.org/pdf/1810.04805.pdf)
* [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)
* [XLM-R](https://ai.facebook.com/blog/-xlm-r-state-of-the-art-cross-lingual-understanding-through-self-supervision/)

NLP systems use contrastive methods by masking or substituting input words. The goal is to reconstruct the original version of a corrupted text. The general technique of training a model to restore a corrupted version of an input is called a **denoising auto-encoder**.

These techniques cannot be easily extended to CV: it is considerably more difficult to represent uncertainty in the prediction for images than it is for words. We cannot list all possible video frames and compute a prediction score (after softmax layer) to each of them, because there is an infinite number of them (high dimensional continuous object vs discrete outcome).

New techniques such as SwAV are starting to beat accuracy records in vision tasks: latest research project [SEER](https://ai.facebook.com/blog/seer-the-start-of-a-more-powerful-flexible-and-accessible-era-for-computer-vision) leverages SwAV and other methods to pretrain a large network on a billion random unlabeled images, showing that self-supervised learning can excel at computer vision tasks as well.

FAIR released a model family called https://arxiv.org/abs/2003.13678 RegNets that are ConvNets capable of scaling to billions (potentially even trillions) of parameters and can be optimized to fit different runtime and memory limitations.


## Energy-based models
* Trainable system that measures the compatibility between an observation x and a proposed prediction y. 
* If x and y are compatible, the energy is a small number; if they are incompatible, the energy is a large number. 

Training an EBM consists of two parts: 
1. Training it to produce low energy for compatible x and y 
2. **Finding a way to ensure that for a particular x, the y values that are incompatible with x produce a higher energy than those that are compatible with x**

The second point is where the difficulty lies

A well-suited architecture is **Siamese networks** or joint embedding architecture:
* two identical copies of the same network
* ine network is fed with x and the other with y and they produce an embedding for x and y respectively
* a third module computes the energy as the distance between the two embedding vectors

Without a specific way to ensure that the networks produce high energy when x and y differ, the two networks could collapse to always produce identical output embeddings.

Two categories of techniques to avoid collapse: **contrastive methods** (method used in NLP) and **regularization methods**.

Another interesting avenue is **latent-variable predictive models**:
* given an observation x the model produces a set of multiple compatible predictions
* as a latent-variable z varies within a set, the output varies over the set of plausible predictions

Latent-variable models can be trained with contrastive methods. A good example is a generative adversarial network:
* the critic (or discriminator) can be seen as computing an energy indicating whether the input y looks good
* the generator network is trained to produce contrastive samples to which the critic is trained to associate high energy.

Contrastive methods have a major issue: they are very inefficient to train. In high-dimensional spaces such as images, there are many ways one image can be different from another.

**“Happy families are all alike; every unhappy family is unhappy in its own way”**
- Leo Tolstoy in Anna Karenina

## Non-contrastive energy-based SSL
Non-contrastive methods applied to joint embedding architectures is the hottest topic in SSL for vision:
* [DeeperCluster](https://openaccess.thecvf.com/content_ICCV_2019/html/Caron_Unsupervised_Pre-Training_of_Image_Features_on_Non-Curated_Data_ICCV_2019_paper.html)
* [ClusterFit](https://arxiv.org/abs/1912.03330)
* [MoCo-v2](https://arxiv.org/abs/2003.04297)
* [SwAV](https://arxiv.org/abs/2006.09882)
* [SimSiam](https://arxiv.org/abs/2011.10566)
* Barlow Twins
* [BYOL](https://arxiv.org/abs/2006.07733) from DeepMind

They use various tricks, such as:
* computing virtual target embeddings for groups of similar images (DeeperCluster, SwAV, SimSiam)
* making the two joint embedding architectures slightly different through the architecture or the parameter vector (BYOL, MoCo)
* Barlow Twins tries to minimize the redundancy between the individual components of the embedding vectors.

Perhaps a better alternative in the long run will be to devise non-contrastive methods with latent-variable predictive models.
The main obstacle is that they require a way to minimize the capacity of the latent variable:
* the volume of the set over which the latent variable can vary limits the volume of outputs that take low energy
* by minimizing this volume, one automatically shapes the energy in the right way

A successful example of such a method is the [Variational Auto-Encoder](https://arxiv.org/abs/1312.6114) (VAE), in which the latent variable is made “fuzzy”, which limits its capacity. But VAE have not yet been shown to produce good representations for downstream visual tasks. 

Another successful example is [sparse modeling](https://www.nature.com/articles/381607a0) but its use has been limited to simple architectures. No perfect recipe seems to exist to limit the capacity of latent variables.

