# Retail Media Networks: Engineering AI-Native Advertising Platforms

This repository contains selected chapters and code examples from a comprehensive guide on engineering Retail Media Networks (RMNs).

## About the Author

**Michael Lei** is a seasoned engineer with 15+ years of experience in Internet Search/Ads, Machine Learning/AI, and Distributed Systems. Key roles include founding member of Google's TeraGoogle project (recognized with Google Founders Award), tech lead on multiple data and machine learning projects in Google Display Ads, and delivering core technology products at eBay, Appen, Meta and Coupang. Holds multiple patents in Internet Ads and Content Recommendations.

## About the Book
Retail Media Networks represent a fundamental shift in digital advertising, leveraging first-party retailer data to deliver highly relevant ads with deterministic attribution. This book explores the engineering architecture, machine learning systems, and real-time decisioning engines required to build a production-grade RMN.

It covers topics ranging from high-performance ad serving and multi-tower deep learning models to auction mechanics, budget pacing, and privacy-preserving data infrastructure.

## Available Chapters
This repository currently hosts the following chapters for public access:

- **[Preface](preface.md)**
  - Overview of the book's motivation, target audience, and organization.
- **[Chapter 2: The Retail Media Network Ecosystem](ch2_rmn_problem_domain.md)**
  - Defines what RMNs are, why they emerged, and what makes them architecturally distinct from traditional networks. Maps the domain—campaign hierarchies, targeting types, bidding mechanics, measurement—so that engineering decisions in later chapters are grounded in business context.
- **[Chapter 5: Ads Retrieval](ch5_ads_retrieval.md)**
  - Technical deep dive into candidate generation strategies, indexing systems (inverted indices, embedding-based retrieval), approximate nearest neighbor search, multi-stage retrieval pipelines, and lightweight scoring models for efficient candidate ranking.
- **[Chapter 6: Multi-Tower Scoring Model](ch6_multi_tower_scoring_model.md)**
  - Detailed exploration of the neural architecture for CTR/CVR prediction, including feature engineering, tower structures, multi-task learning, and training strategies.
- **[Appendix 2: Embedding Architectures for Sequential Features](ap2_embeddings.md)**
  - Technical foundations for embedding-based feature encoding, covering sequential encoders and advanced techniques for representing user and product data.

## Code & Notebooks
Practical implementations and experiments are available in the [`code/`](https://github.com/thunderbird2009/book_rmn/tree/main/code) directory:

- **[Multi-Tower Scoring Model (Notebook)](https://github.com/thunderbird2009/book_rmn/blob/main/code/ch6_multi_tower_scoring_model.ipynb)**: A hands-on implementation of the scoring model discussed in Chapter 6.
- **[Embeddings Test (Notebook)](https://github.com/thunderbird2009/book_rmn/blob/main/code/ap2_embeddings_test.ipynb)**: Experiments with embedding architectures.

## Full Table of Contents (Preview)
The complete work includes:

**Part I — Foundations**
- Ch 1: Traditional Ad Networks
- Ch 2: The RMN Problem Domain (Public Access)

**Part II — Core Serving Stack**
- Ch 3: Ad Serving Architecture
- Ch 4: Auction & Cost Tracking
- Ch 5: Ads Retrieval (Public Access)
- Ch 6: Multi-Tower Scoring Model (Public Access)
- Ch 7: Budget Pacing & Spend Control

**Part III — Media Planning & Optimization**
- Ch 8: Automated Creative Generation
- Ch 9: Creative Selection and Learning
- Ch 10: Predictive Audiences

**Part IV — Trust, Measurement & Governance**
- Ch 11: Ad Fraud Detection and Prevention
- Ch 12: Measurement, Attribution, and Incrementality
- Ch 13: Privacy, Governance & Identity Resolution

**Part V — Appendices**
- Appendix 1: AdIndex System Implementation
- Appendix 2: Embedding Architectures for Sequential Features (Public Access)
- Appendix 3: Identity Solution Implementations

**Back Matter**
- [Glossary](glossary.md)
