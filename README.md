# Book RMN: Building Retail Media Networks

This repository contains selected chapters and code examples from a comprehensive guide on engineering Retail Media Networks (RMNs).

## About the Book
Retail Media Networks represent a fundamental shift in digital advertising, leveraging first-party retailer data to deliver highly relevant ads with deterministic attribution. This book explores the engineering architecture, machine learning systems, and real-time decisioning engines required to build a production-grade RMN.

It covers topics ranging from high-performance ad serving and multi-tower deep learning models to auction mechanics, budget pacing, and privacy-preserving data infrastructure.

## Available Chapters
This repository currently hosts the following chapters:

- **[Chapter 2: Ad Serving Architecture](ch2_ad_serving_architecture.md)**
  - Deep dive into the end-to-end request flow, system components (Ad Server, Retrieval, Scoring, Auction), and latency constraints of a sub-100ms ad serving pipeline.
- **[Chapter 5: Multi-Tower Scoring Model](ch5_multi_tower_scoring_model.md)**
  - Detailed exploration of the neural architecture used for CTR/CVR prediction, including feature engineering, tower structures, and training strategies.
- **[Appendix 5: Embeddings](ap5_embeddings.md)**
  - Technical foundations for embedding-based feature encoding, covering sequential encoders and advanced techniques for representing user and product data.

## Code & Notebooks
Practical implementations and experiments are available in the `code/` directory:

- **[Multi-Tower Scoring Model (Notebook)](https://github.com/thunderbird2009/book_rmn/blob/main/code/ch5_multi_tower_scoring_model.ipynb)**: A hands-on implementation of the scoring model discussed in Chapter 5.
- **[Embeddings Test (Notebook)](https://github.com/thunderbird2009/book_rmn/blob/main/code/ap5_embeddings_test.ipynb)**: Experiments with embedding architectures.

## Full Table of Contents (Preview)
The complete work includes:
- Ch 0: Traditional Ad Networks
- Ch 1: The RMN Ecosystem
- Ch 2: Ad Serving Architecture (Included)
- Ch 3: Auction & Cost Tracking
- Ch 4: Ads Retrieval
- Ch 5: Multi-Tower Scoring Model (Included)
- Ch 6: Budget Pacing & Spend Control
- ...and more on Measurement, Data Privacy, and MLOps.


