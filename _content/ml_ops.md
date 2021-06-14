---
layout: post
title: MLOps
tag: MLOps
---
Apply DevOps principles to ML systems (MLOps)

Pitfalls in operating ML-based systems in production: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43146.pdf

[source](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

## DevOps versus MLOps
DevOps: developing and deploying large-scale software. Shortening development cycles and dependable (reliable) releases

* **continuous integration (CI)**: frequently merging all developer's branch onto master branch. used in combination with automated unit tests.
* **continuous delivery (CD)**: build, test, release software in incremental updates to increase frequency and reliability of release. Repeatable deployment process is needed (continuous *deployment* refers to automated deployment).

Challenges of applying DevOps to ML:
* **team**: data scientists and ML researchers don't have the skills to build production-class services
* **development**: ML is experimental in nature. Tracking results and maintaining reproducibility (while maximizing code reusability, i.e. building software knowledge for other projects).
* **testing**: typical unit and integration tests + data validation and model evaluation
* **deployment**: multi-step pipeline to automatically retrain and deploy model
* **production**: monitor data drift and online model performance.

In MLOps:
* CI is about testing code + testing data, data schemas and models.
* CD is about an ML training pipeline deploying a model prediction service
* CT (continuous training) is about automatically retraining and serving models

## Data science steps for ML
After **defining business use case and success criteria**:
data extraction, EDA, data prep, model training, model evaluation, model validation (adequate for deployment? performance better than baseline?), model serving (microservices with REST API to serve online predictions / embedded model to mobile device), model monitoring.

Disconnect between ML and operations can lead to differences between inference and test performance due to different training and serving pipelines. It's important to have the **same pipeline for development and production**.
