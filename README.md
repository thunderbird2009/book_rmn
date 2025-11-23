# Building Retail Media Networks: Architecture, Algorithms, and Machine Learning

This repository contains selected chapters and code examples from a comprehensive guide on engineering Retail Media Networks (RMNs).

## About the Author

**Michael Lei** is a seasoned engineer with 15+ years of experience in Internet Search/Ads, Machine Learning/AI, and Distributed Systems. Key roles include founding member of Google's TeraGoogle project (recognized with Google Founders Award), tech lead on multiple data and machine learning projects in Google Display Ads, and delivering core technology products at eBay, Appen, Meta and Coupang. Holds multiple patents in Internet Ads and Content Recommendations.

## About the Book
Retail Media Networks represent a fundamental shift in digital advertising, leveraging first-party retailer data to deliver highly relevant ads with deterministic attribution. This book explores the engineering architecture, machine learning systems, and real-time decisioning engines required to build a production-grade RMN.

It covers topics ranging from high-performance ad serving and multi-tower deep learning models to auction mechanics, budget pacing, and privacy-preserving data infrastructure.

## Available Chapters
This repository currently hosts the following chapters:

- **[Chapter 1: RMN Problem Domain](ch1_rmn_problem_domain.md)**
  - Introduction to the Retail Media Network problem domain, the ecosystem, key stakeholders (retailers, advertisers, consumers), and the unique value proposition of first-party data in advertising.
- **[Chapter 2: Ad Serving Architecture](ch2_ad_serving_architecture.md)**
  - Deep dive into the end-to-end request flow, system components (Ad Server, Retrieval, Scoring, Auction), and latency constraints of a sub-100ms ad serving pipeline.
- **[Chapter 3: Auction and Cost Tracking](ch3_auction_and_cost_tracking.md)**
  - Comprehensive coverage of auction mechanics (first-price, second-price, VCG), bid ranking strategies, cost calculation methods, and real-world auction design trade-offs.
- **[Chapter 4: Ads Retrieval](ch4_ads_retrieval.md)**
  - Technical deep dive into candidate generation strategies, indexing systems (inverted indices, embedding-based retrieval), approximate nearest neighbor search, multi-stage retrieval pipelines, and lightweight scoring models for efficient candidate ranking.
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
- Ch 1: The RMN Problem Domain (Included)
- Ch 2: Ad Serving Architecture (Included)
- Ch 3: Auction & Cost Tracking (Included)
- Ch 4: Ads Retrieval (Included)
- Ch 5: Multi-Tower Scoring Model (Included)
- Ch 6: Budget Pacing & Spend Control
- ...and more on Measurement, Data Privacy, and MLOps.


