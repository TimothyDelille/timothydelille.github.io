---
layout: post
title: TextRank
tag: Natural Language Processing
author: University of Texas, 2004
---
* TOC
{:toc}

* unsupervised graph-based ranking model for text processing
* applied to two tasks: (1) extract keyphrases representative for a given text and (2) sentence extraction task (most important sentences in a text), used to build extractive summaries

## Introduction
A graph-based ranking algorithm is a way of deciding on the importance of a vertex within a graph, by taking into account global information recursively computed from the entire graph, rather than relying only on local vertex-specific information.

Applications: automated extraction of keyphrases, extractive summarization and word sense disambiguation

## TextRank model
* when one vertex links to another, it is basically casting a vote for that other vertex
* the importance of the vertex casting the vote determines how important the vote itself is

Let $$G = (V, E)$$ be a directed graph. Let $$In(V_i)$$ be the set of vertices that point to $$V_i$$ (predecessors) and $$Out(V_i)$$ those that $$V_i$$ points to (successors). By Google's PageRank (Brin and Page, 1998):
$$score(V_i) = (1-d) + d\sum_{j\in In(V_i)} \frac{1}{\lvert Out(V_j) \rvert} score(V_j)$$

$$d$$ is a damping factor between 0 and 1 (set to 0.85) and represents the probability of jumping from a given vertex to another random vertex in the graph ("random surfer model") where a user clicks on links at random with a probability $$d$$ and jumps to a completely new page with probability $$1 - d$$.

Starting from arbitrary score values, computation iterates until convergence. Convergence is achieved when error rate for any vertex falls below given threshold. Error rate is defined as difference between "real" score $$S(V_i)$$ and score computed at iteration $$k$$, $$S^k (V_i)$$. Since "real" score is not known a priori, it is approximated with $$S^{k+1}(V_i) - S^k(V_i)$$.

As number of edges increases, converge is achieved after fewer iterations.

For undirected graphs, in-degree of a vertex is equal to the out-degree of the vertex.

### Weighted Graphs
Incorporate strength (due to multiple or partial links between vertices) of the connection between two vertices $$V_i$$ and $$V_j$$ as weight $$w_ij$$.

Formula becomes:

$$score(V_i) = (1-d) + d\sum_{V_j\in In(V_i)} \frac{w_ji}{\sum_{V_k \in Out(V_i)} w_{jk}} score(V_j)$$

### Text as a Graph
Depending on application:
* text units can have various sizes: words, collocations, sentences, ...
* type of relations: lexical, semantic, contextual overlap, ...

## Keyword Extraction
### Approach
* co-occurence relation, controlled by the distance between word occurences (two vertices are connected if the lexical units co-occur within a window of maximum $$2 \leq N \leq 10$$ words)
* to avoid excessive growth of graph size, only consider single words
* vertices are restricted with syntactic filters using part of speech tags: best results observed by considering nouns and adjectives only for addition to the graph
* after convergence, top $$T = \frac{\lvert V \rvert}{3}$$ vertices are retained for post-processing (sequences of adjacent key-words are collapsed into a multi-word keyword)

### Results
* the larger the window, the lower the precision (relation between words that are further apart is not strong enough to define a connection in the text graph).
* linguistic information such as part of speech tags helps the process of keyword extraction
* results obtained with directed graphs are worse then results obtained with undirected graphs: despite a natural flow in running text, there is no natural direction that can be established co-occuring words

## Sentence Extraction
* same as keyword extraction except text units are entire sentences
* used in automatic summarization
* a vertex is added to the graph for each sentence in the text
* *co-occurence* is not a meaningful relationship for large contexts
* instead we use *similarity* measured as a function of their content overlap (number of common tokens / or count words of certain syntactic categories only)
* normalization factor to avoid promoting long sentences: divide the content overlap with the length of each sentence

$$similarity(S_i, S_j) = \frac{\lvert \{ w_k \vert w_k \in S_i\cap S_j \} \rvert}{\log (\lvert S_i \rvert) + \log(\lvert S_j \rvert)}$$

* other sentence similarity measures: string kernels, cosine similarity, longest common subsequence
* graph is weighted with strength of the association
* evaluated using ROUGE (published after BLEU for machine translation), which is based on ngram statistics, found to be highly correlated with human evaluations

## Notes
* *knitting phenomenon*: words aree shared in different parts of the discourse and serve to knit the discourse together
* *text surfing*: (text cohesion) from a certain concept in a text, we are likely to follow links to connected concepts (lexical or semantic relation)
* sentences / words that are highly recommended by other units in the text are likely to be more informative for the given text
