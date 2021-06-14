---
layout: post
title: Initializing Neural Networks
published: False
---
https://www.deeplearning.ai/ai-notes/initialization/#:~:text=Initializing%20all%20the%20weights%20with,the%20same%20features%20during%20training.&text=Thus%2C%20both%20neurons%20will%20evolve,neurons%20from%20learning%20different%20things.

http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi

https://arxiv.org/pdf/1502.01852.pdf

Karpathy's comments on initialization:

* when regressing some values that have mean $$\mu$$, initialize the output bias to $$\mu$$.
* in classification, if dataset has an imbalance ratio of 1:10, set the bias on the logits such that the network predicts probability of 0.1 at initialization ($$\text{softmax}(z, 1) = \frac{e^{b_1}}{e^b_0 + e^b_1} = 0.1 \Leftrightarrow b_1 = \dots$$)
* this will eliminate "hockey stick" loss curves (neural net is learning the bias during the first iterations) and speed up convergence
* [source](https://karpathy.github.io/2019/04/25/recipe/)
