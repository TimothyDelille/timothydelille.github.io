---
layout: post
title: Stanford CS329S: Machine Learning Systems Design
author: Chip Huyen (2021)
---
## ML in research vs production: misalignment of interests
* objective: state-of-the-art (SOTA) at the cost of complexity
* e.g. ensembling is popular to win competitions but increases complexity
* risks of complexity: more error-prone to deploy, slower to serve, harder to interpret
* benchmarks incentivize accuracy at the expense of compactness, fairness, energy efficiency, interpretability
* research prioritizes **fast training** (high throughput), production prioritizes **fast inference** (low latency)

Without batching, higher latency means lower throughput. With batching, higher throughput means higher latency.

In 2009, Google showed increasing web search latency 100 to 400 ms reduces daily number of searches per user by 0.2% to 0.6%.
In 2019, Booking.com found that an increase of about 30% in latency cost about 0.5% in conversion rates.

Real-life data is streaming, shifting, sparse, imbalanced, incorrect, private, biased...

Interpretability enables detecting biases and debugging a model.

Majority of ML-related jobs are in productionizing ML as off-the-shelf models become more accessible and the "bigger, better" approach the research community is taking requires short-term business applications (tens of millions of dollars in compute alone).

Applications developped with the most/best data win.

## Challenges in ML production
* Data testing: is sample useful ?
* Data and model versioning: see [DVC](https://github.com/iterative/dvc)
* Monitoring for data-drift: see [Dessa](https://www.dessa.com/) (acquired by Square)
* Data labeling: see [Snorkel](https://www.snorkel.org/)
* CI/CD test: see [Argo](https://argoproj.github.io/)
* Deployment: see [OctoML](https://octoml.ai/)
* Model compression (e.g. to fit onto consumer devices): see [Xnor.ai](#), acquired by Apple for ~$200M.
* Inference optimization: speed up inference time by fusing operations together, using lower precision, making a model smaller. See [TensorRT](https://developer.nvidia.com/tensorrt)
* Edge device: Hardware designed to run ML algorithms fast and cheap. Example: [Coral SOM](https://coral.ai/products/som/)
* Privacy: GDPR-compliant (General Data Protection Regulation)? See [PySyft](https://github.com/OpenMined/PySyft)
* Data manipulation: see [Dask](https://github.com/dask/dask) (parallel computation in Python, mimicking pandas)
* Data format: row-based data formats like CSV require to load all features even if using a subset of them. Columnar file formats like PARQUET and ORC are optimized for that use case.

## ML systems design
Defining interface, algorithms, data, infrastructure and hardware.

Many cloud services enable autoscaling the number of machines depending on usage.

Subject matter experts (auditors, bankers, doctors, lawyers etc...) are overlooked developers of ML systems. We only think of them to label data but they are useful for: problem formulation, model evaluation, developping user interface...

### Online prediction (a.k.a. HTTP prediction) vs batch prediction
* **batch prediction**:
  * asynchronous
  * periodical
  * high throughput
  * processing accumulated data when you don’t need immediate results (e.g. recommendation systems)
* **online prediction**: instantaneous (e.g. autocomplete)

**Batch prediction is a workaround for when online prediction isn’t cheap enough or isn’t fast enough**

### Edge computing vs cloud computing
Edge computing: computation done on the edge (= on device) as opposed to cloud computing (on servers).

Cloud computing is used when ML model requires too much compute and memory to be run on device.

Disadvantages of cloud computing:
* network latency is a bigger bottleneck than inference latency.
* storing data of many users in the same place means a breach can affect many people
* servers are costly

**The future of ML is online and on-device** (+ see federated learning for training over edge devices).

### Online learning vs offline learning
Data becomes available sequentially vs in batch. E.g. Ordinary Least Squares vs Recursive Least Squares.
