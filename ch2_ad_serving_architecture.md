# Chapter 2: Ad Serving Architecture for Retail Media Networks (RMNs)

## 1. Introduction
* The Strategic Role of Ad Serving in RMNs.
* How RMN Ad Serving Differs from Traditional Ad Networks (references `ch0_traditional_ad_network.md`).
* Core Business Objectives:
    * Enhancing the Shopper Experience with relevant ads.
    * Maximizing Return on Investment (ROI) for Advertisers.
    * Driving Incremental Revenue for the Retailer.

## 2. Core Requirements for an RMN Ad Server
* **Functional Requirements:**
    * **Ad Request Handling:** Processing requests from various placements (e.g., homepage, search results, product pages).
    * **Targeting & Audience Segmentation:** Leveraging rich 1st-party data for precise audience targeting.
    * **Campaign & Budget Management:** Tools for advertisers to manage their spend and campaign lifecycle.
    * **Ad Retrieval & Ranking:** Finding and scoring the most relevant ads for a given opportunity.
    * **Auction & Ad Selection:** Running an auction to select the winning ad(s).
    * **Reporting & Analytics:** Providing performance metrics to advertisers.
* **Non-Functional Requirements (The Engineering Challenge):**
    * **Low Latency:** Strict Service Level Objectives (SLOs) to avoid impacting the shopper experience.
    * **High Scalability:** Handling massive traffic volumes (high Queries Per Second).
    * **Relevance:** The core principle balancing shopper needs and advertiser goals.
    * **Extensibility:** Architecture must evolve to support new ad formats, ranking models, and features.
    * **Data Privacy & Security:** Securely managing and utilizing sensitive first-party data.

## 3. High-Level Ad Serving Flow: A Shopper's Journey
This section will walk through the primary use case of a shopper visiting the retail site.
1.  **Ad Request:** A shopper's browser requests a page, triggering a call to the Ad Server for one or more ad slots. The request contains context like the page URL, user identifiers, and keywords.
2.  **Enrichment:** The Ad Server enriches the request with more user information (e.g., past purchase behavior, segment data).
3.  **Candidate Generation (Retrieval):** The system quickly retrieves a set of all eligible ads that match the basic targeting criteria. This is a crucial first filter to reduce the search space. (This will be detailed in `ch3_ads_retrieval_model.md`).
4.  **Prediction & Ranking:** Machine learning models predict the likelihood of clicks, conversions, etc., for each candidate ad. The ads are then scored and ranked based on these predictions and advertiser bids. (The models for this are discussed in `ch4_multi_tower_architecture.md`).
5.  **Auction & Selection:** An auction is held among the top-ranked ads to select the final winner(s), considering factors like budget and pacing.
6.  **Ad Response & Rendering:** The winning ad(s) are returned to the browser to be displayed to the shopper.
7.  **Logging & Tracking:** Impressions, clicks, and other events are logged for reporting, billing, and model training.

## 4. High-Level System Components
* **Online Systems (Real-time & Low-Latency):**
    * **Ad Server:** The central orchestrator of the ad serving flow.
    * **Ad Retrieval Service:** Fetches ad candidates from an Ad Index.
    * **Ad Ranking Service:** Applies ML models to score the candidates.
    * **User Data Service:** Provides real-time user profile and segment information.
* **Offline Systems (Batch & Near-Real-time):**
    * **Data Pipelines (ETL/ELT):** Process logs and events for reporting and feature engineering.
    * **ML Model Training Platform:** Continuously trains and updates the ranking models.
    * **Campaign Database:** The source of truth for all advertiser campaign settings.
    * **Reporting & Analytics Engine:** Aggregates data to populate advertiser dashboards.

## 5. Summary
* Recap of the ad serving flow and its importance.
* A look ahead to the deeper dives in subsequent chapters on retrieval (`ch3`) and ranking (`ch4`).