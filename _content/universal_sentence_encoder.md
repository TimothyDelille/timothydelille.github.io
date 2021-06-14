---
layout: post
title: Universal Sentence Encoder
tag: Natural Language Processing
author: Google Research - 2018
---
* TOC
{:toc}

* Encoding sentences into embedding vectors for transfer learning to other NLP tasks
* Transfer learning using sentence embeddings tends to outperform word level transfer

Two variants: **transformer encoder** and **deep averaging network**

## Transformer Encoder
* targets **high accuracy** at the cost of **greater model complexity**
* input is lowercased Penn Treebank tokenized string (see J&M chap. 2.4)
* compute context aware representations of words
* compute element-wise sum at each word position to get fixed length sentence encoding vector
* divide by square root of sentence length to mitigate length effects
* time complexity is $$O(n^2)$$ / space complexity is $$O(n^2)$$

## Deep Averaging Network (DAN)
(Iyyer et al., 2015)
* targets **efficient inference** with slightly **reduced accuracy**
* input embeddings for words and bi-grams are first averaged together
* and then passed through a feedforward deep neural network to produce sentence embeddings
* time complexity is $$O(n)$$ / space complexity is $$O(1)$$
* space complexity is dominated by the parameters used to store unigram and bigram embeddings (for short sentences, can be twice the memory usage of transformer model)

## Multi-task learning
single encoding model used to feed multiple downstream tasks
* skip-thought task (unsupervised from running text)
* conversational input-response task
* classification tasks (supervised)

## Transfer Learning
* For *sentence classification* transfer tasks, the output of the *transformer* and *DAN sentence encoders* are provided to a task specific deep neural net.
* For *pairwise semantic similarity* task, compute cosine similarity of the sentence embeddings produced by *both* encoder variants and then apply arccos to convert into angular distance (performs better than raw cosine similarity) $$\text{sim}(u,v) = 1-\arccos \big( \frac{u\cdot v}{\lVert u\rVert \lVert v \rVert} \big)/\pi$$

## Results
* hyperparemeters tuned using [Google Vizier](https://research.google/pubs/pub46180/)
* assess bias in encoding models by evaluating strength of associations learned on WEAT word lists (Word Embeddings Association Test)
* sentence-level + word-level (word2vec skip-gram embeddings) transfer > sentence-level transfer only > word-level transfer only
* transfer learning is most critical when training data for a target task is limited
* as training set size increases, marginal improvement of using transfer learning decreases

Note: switching to [Sentence Piece](https://github.com/google/sentencepiece) vocabulary instead of words significantly reduces vocabulary size, which is a major contributor of model sizes (good for on-device or browser-based implementations)
