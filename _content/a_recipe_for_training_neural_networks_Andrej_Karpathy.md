---
layout: post
title: A Recipe for Training Neural Networks
author: Andrej Karpathy
published: False
---

## Notes
* libraries and frameworks give a false impression of plug and play
* reality is a network does not magically work; batch norm does not magically make it converge faster; just because you can formulate a problem as RL doesn't mean you should
* "possible error surface" is large, logical (as opposed to syntactic); neural nets fail silently
* network can still work relatively well despite errors:
  - forgot to flip labels when flipping images left-right during data augmentation
  - autoregressive model takes as input the word it's trying to predict due to an off-by-one bug
  - tried to clip gradients but clipped loss instead, causing outliers to be ignored
  - initialized weights from pre-trained checkpoint but didn't use original mean
  - used `view` instead of `transpose`/`permute` and screwed up broadcasting
* build from simple to complex and validate concrete hypotheses at every step of the way
* do not introduce a lot of "unverified" complexity at once

## 1. become one with the data
* do not touch any neural net code
* scan through thousands of examples (takes hours), understand their **distribution**, look for **patterns**, look for **data imbalance** and **biases**
* start thinking about possible architectures: e.g. are very local features enough or do we need global context? does spatial position matter or can we average pool it out?
* neural net is effectively a **compressed version of your dataset**: if your network gives you (mis)predictions that are not consistent with what you have seen in the data, something is off
* search/filter/sort by attributes and visualize their distribution and the outliers
* **outliers almost always uncover some bugs in data quality or preprocessing**

## 2. set up end-to-end training/evaluation skeleton + get dumb baselines
* pick some simple model you couldn't possible have screwed up somehow (e.g. linear classifier or tiny ConvNet), train it, visualize the losses, other metrics, model predictions and perform a series of ablation experiments with explicit hypotheses.
* **fix random seed** for reproducibility
* **simplify**: no data augmentation or regularization
* **add significant digits to your eval**: correctness comes first
* **verify loss at initialization**: e.g. you should measure $$-\log \frac{1}{n_{classes}}$$ on a softmax at initialization (can derive default values for L2 regression, Hubert loss etc...)
* **init well**: e.g. if regressing values with mean 50, then initialize final bias to 50. If imbalanced dataset of ratio 1:10, set bias on logits such that network predicts probability 0.1 at init. This will eliminate "hockey stick" loss curves where network basically learns the biases during first few iterations.
* **get human baseline** (predict dataset manually / get your own accuracy)
* **input-dependent baseline**
