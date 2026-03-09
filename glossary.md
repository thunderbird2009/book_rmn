# Glossary

This glossary provides concise definitions of the key technical terms, acronyms, and concepts used throughout the book. Each entry includes cross-references to the chapters where the term is discussed in detail. For introductory definitions of core RMN domain terms, see also **Chapter 2, Section 2.2**.

---
<details>
<summary>Table of Contents</summary>

- [A](#a)
- [B](#b)
- [C](#c)
- [D](#d)
- [E](#e)
- [F](#f)
- [G](#g)
- [H](#h)
- [I](#i)
- [K](#k)
- [L](#l)
- [M](#m)
- [N](#n)
- [O](#o)
- [P](#p)
- [Q](#q)
- [R](#r)
- [S](#s)
- [T](#t)
- [U](#u)
- [V](#v)
- [W](#w)

</details>

---

## A

**A/B Test** — A controlled experiment that randomly assigns users to a control group (existing experience) and a treatment group (new variant) to measure causal impact. Used throughout the ad-serving stack for validating model changes, auction rule updates, and creative variants. (Ch 9, Ch 12)

**Ad Group** — A grouping of ads within a campaign that share the same targeting criteria and bid settings. Individual ads within an ad group participate in auctions as distinct candidates. (Ch 2, Ch 4)

**Ad Index (AdIndex)** — A data structure that stores and indexes all active ad candidates for fast retrieval during ad serving. Typically built on inverted indices with support for real-time updates. (Ch 3, Ch 5, Ap 1)

**Ad Server** — The central service that orchestrates the real-time ad request pipeline: receiving a request, calling retrieval and scoring services, running the auction, selecting the winner, assembling the creative, and logging the event. (Ch 3)

**ANN (Approximate Nearest Neighbor)** — A family of algorithms that find approximate (rather than exact) nearest neighbors in high-dimensional vector spaces, trading a small amount of recall for orders-of-magnitude speedup. Key implementations include FAISS, ScaNN, and HNSW. (Ch 5, Ap 1)

**Attribution** — The process of assigning credit for a conversion to one or more prior ad interactions (impressions, clicks). RMNs benefit from deterministic attribution via first-party data, unlike traditional networks that rely on probabilistic methods. (Ch 2, Ch 12)

**Attribution Window** — The time period after an ad interaction during which subsequent conversions are credited to that interaction. Typical windows range from 1 to 30 days. (Ch 2, Ch 12)

**Auction** — The mechanism by which ad candidates compete for an impression. The ad server evaluates bids, predicted engagement, and quality scores to determine the winner and the price paid. (Ch 4)

**Auto-Bidding** — An automated bidding strategy where the platform adjusts bids per auction to optimize toward an advertiser-specified goal (e.g., Target ROAS, Target CPA, Maximize Conversions). (Ch 4, Ch 7)

---

## B

**Bandit Algorithm** — An online learning algorithm that balances exploration (trying less-known options) with exploitation (choosing the current best option). Used for creative selection, audience exploration, and dynamic optimization. Common variants include Thompson Sampling and Upper Confidence Bound (UCB). (Ch 9)

**Bid Landscape** — A function mapping hypothetical bid values to expected outcomes (impressions, clicks, conversions), used by auto-bidding systems to compute optimal bids. (Ch 1, Ch 4)

**Boolean Targeting** — A retrieval method that matches ads to requests using boolean logic over targeting attributes (keywords, audience segments, product categories). Implemented via inverted indices. (Ch 5, Ap 1)

**Budget Pacing** — The process of distributing an advertiser's budget over time (e.g., evenly across a day or flight) to avoid premature exhaustion. Techniques include PID controllers, throttling, and Lagrangian optimization. (Ch 2, Ch 7)

---

## C

**Calibration** — The property that a model's predicted probabilities match observed frequencies. A well-calibrated pCTR model predicting 5% click-through rate should see approximately 5 clicks per 100 impressions. Critical for fair auction ranking and accurate billing. (Ch 6)

**Campaign** — The top-level advertising entity containing one or more ad groups. Defines overall budget, flight dates, and optimization objectives. (Ch 2)

**Cannibalization** — Conversions that would have occurred organically (without ad exposure) but are incorrectly attributed to ads. Incrementality testing aims to measure and correct for this effect. (Ch 2, Ch 12)

**CCPA (California Consumer Privacy Act)** — California state law granting consumers rights over their personal data, including the right to know what data is collected, to delete it, and to opt out of its sale. (Ch 13)

**Clean Room** — A secure computation environment where multiple parties (e.g., retailer and advertiser) can perform joint analyses on combined data without either party accessing the other's raw records. (Ch 2, Ch 10, Ch 13)

**Click-Through Rate** — See [**CTR**](#c).

**Contextual Bandit** — A bandit algorithm that conditions its selection policy on contextual features (user attributes, page context) rather than treating each arm identically across all contexts. Used for personalized creative selection. (Ch 9)

**Conversion** — A desired action taken by a user after ad exposure, such as a purchase, add-to-cart, or sign-up. The primary outcome metric for most RMN campaigns. (Ch 2, Ch 12)

**Cost Per Acquisition** — See [**CPA**](#c).

**Cost Per Click** — See [**CPC**](#c).

**Cost Per Mille** — See [**CPM**](#c).

**CPA (Cost Per Acquisition)** — The cost to the advertiser for each conversion: `ad_spend / conversions`. A key efficiency metric for performance campaigns. (Ch 2, Ch 4)

**CPC (Cost Per Click)** — A pricing model where the advertiser pays only when a user clicks the ad. The dominant pricing model in search and sponsored product advertising. (Ch 2, Ch 4)

**CPM (Cost Per Mille)** — A pricing model where the advertiser pays per 1,000 impressions served. Common in display and video advertising. (Ch 2, Ch 4)

**Creative** — The visual or textual asset shown to the user: a product image, banner, video, headline, or body text. May be static (pre-built) or dynamically assembled. (Ch 2, Ch 8, Ch 9)

**Cross-Feature Interaction** — A neural network component that explicitly models interactions between features from different towers (e.g., user-ad, user-context). Techniques include cross networks (DCN), factorization machines (DeepFM), and attention mechanisms. (Ch 6)

**CTR (Click-Through Rate)** — The ratio of clicks to impressions: `clicks / impressions`. A primary engagement metric used in ranking and billing. (Ch 1, Ch 6)

**CVR (Conversion Rate)** — The ratio of conversions to clicks (post-click CVR) or to impressions (post-view CVR): `conversions / clicks`. Used alongside CTR for multi-task prediction. (Ch 1, Ch 6)

---

## D

**DCN (Deep & Cross Network)** — A neural architecture that uses explicit cross layers to model feature interactions alongside a deep network for implicit interactions. Used in ranking models. (Ch 6)

**DCO (Dynamic Creative Optimization)** — The process of automatically assembling and selecting creative elements (headline, image, CTA, layout) in real time based on user context and learned performance. (Ch 8, Ch 9)

**DeepFM** — A neural ranking model combining factorization machines (for second-order feature interactions) with a deep network (for higher-order interactions). (Ch 6)

**Deterministic Attribution** — Attribution that links ad exposure to conversion using the same logged-in user ID, providing high-confidence credit assignment. A key advantage of RMNs over cookie-based networks. (Ch 2, Ch 12)

**Differential Privacy** — A mathematical framework for quantifying and limiting the privacy loss when releasing aggregate statistics computed from individual-level data. Achieved by adding calibrated noise. (Ch 13)

**DIN (Deep Interest Network)** — A neural architecture that uses attention mechanisms to model a user's diverse interests by weighting historical behaviors relative to the candidate ad. (Ch 6)

**DIEN (Deep Interest Evolution Network)** — An extension of DIN that adds a GRU-based sequential model to capture the temporal evolution of user interests. (Ch 6)

**DNF (Disjunctive Normal Form)** — A boolean expression format (OR of ANDs) used to represent ad targeting rules in inverted-index retrieval. Each ad's targeting criteria are expressed as a DNF clause for efficient matching. (Ch 5, Ap 1)

---

## E

**eCPM (Effective Cost Per Mille)** — A unified ranking metric that combines bid and predicted engagement to compare ads on different pricing models: `eCPM = bid × pCTR × 1000` (for CPC ads). Enables apples-to-apples comparison across CPC and CPM ads in the same auction. (Ch 2, Ch 4)

**Embedding** — A learned dense vector representation of a discrete entity (user, ad, product, query) in a continuous vector space. Entities with similar characteristics are mapped to nearby points. Embeddings are the foundation of semantic retrieval and neural ranking. (Ch 5, Ch 6, Ap 2)

**ESMM (Entire Space Multi-Task Model)** — A multi-task learning framework that models CVR over the entire impression space (not just clicked impressions) by decomposing pCVR = pCTR × pCTCVR, addressing sample selection bias. (Ch 6)

**EUID (European Unified ID)** — The European variant of UID2, adapted for GDPR compliance with stricter consent requirements and EU-based operator infrastructure. (Ch 13, Ap 3)

**Exploration** — In online learning, the deliberate selection of less-certain options (ads, creatives, audiences) to gather information and reduce uncertainty, as opposed to exploitation. (Ch 9)

---

## F

**FAISS (Facebook AI Similarity Search)** — An open-source library for efficient similarity search and clustering of dense vectors. Supports multiple index types (IVF, PQ, HNSW) for ANN retrieval at scale. (Ch 5, Ap 1)

**Feature Store** — A centralized system for storing, managing, and serving pre-computed features for ML models. Provides both batch features (for training) and low-latency online features (for inference). (Ch 3, Ch 6)

**First-Party Data** — Data collected directly by the retailer from its own customers: search queries, browsing behavior, purchase history, and loyalty information. The foundational data asset for RMNs. (Ch 2, Ch 10)

**Flight** — The scheduled time period during which a campaign or ad group is active and eligible to serve. (Ch 2)

**FTRL-Proximal (Follow-The-Regularized-Leader)** — An online learning algorithm for training linear models with L1 regularization, producing sparse models suitable for high-dimensional feature spaces. Widely used in traditional ad CTR prediction. (Ch 1)

**Fusion Layer** — The neural network component that combines embeddings from multiple towers (user, ad, context) into a joint representation for final prediction. Techniques include concatenation, element-wise product, and attention-based fusion. (Ch 6)

---

## G

**GDPR (General Data Protection Regulation)** — EU regulation governing the collection, processing, and storage of personal data. Requires explicit consent, data minimization, and the right to erasure. (Ch 13)

**GIVT (General Invalid Traffic)** — Non-human traffic that can be identified through routine means such as known data-center IP lists, declared bots, and user-agent filtering. Contrasted with SIVT. (Ch 11)

**GSP (Generalized Second-Price) Auction** — An auction mechanism where the winner pays the second-highest bid plus a minimum increment (typically $0.01). The dominant auction format in search and sponsored-product advertising. (Ch 1, Ch 2, Ch 4)

---

## H

**HNSW (Hierarchical Navigable Small World)** — A graph-based ANN index that builds a multi-layer proximity graph for fast approximate nearest neighbor search with high recall. (Ch 5, Ap 1)

---

## I

**ID5** — A privacy-focused identity solution that generates partner-specific IDs using probabilistic and deterministic signals without relying on third-party cookies. (Ch 13, Ap 3)

**Impression** — A single instance of an ad being served. Distinctions include: *served* (ad server returned the ad), *rendered* (displayed on screen), and *viewable* (50%+ visible for 1 second or more per MRC standard). (Ch 2, Ch 12)

**Incrementality** — The causal lift in conversions attributable to ad exposure, measured by comparing a treatment group (exposed to ads) against a holdout group (not exposed). Answers: "How many additional conversions did the ads cause?" (Ch 2, Ch 12)

**Inverted Index** — A data structure mapping attribute values (keywords, categories, audience segments) to the set of ads that target those values. The core structure enabling boolean-targeting-based retrieval. (Ch 5, Ap 1)

**IVF (Inverted File Index)** — A vector quantization technique that partitions the embedding space into clusters (Voronoi cells) and searches only the nearest clusters, reducing ANN search cost. Often combined with PQ. (Ch 5, Ap 1)

---

## K

**k-Anonymity** — A privacy property ensuring that each individual in a released dataset is indistinguishable from at least k-1 other individuals with respect to certain identifying attributes. (Ch 13)

**Keyword Targeting** — Matching ads to user search queries based on advertiser-specified keywords with match types: exact match, phrase match, and broad match. (Ch 2, Ch 5)

---

## L

**Lagrangian Pacing** — A budget pacing method that uses Lagrange multipliers to solve the constrained optimization problem of maximizing total value (clicks, conversions) subject to a budget constraint. The multiplier acts as a bid modifier that throttles spend. (Ch 7)

**Lookalike Audience** — An audience segment constructed by finding users who resemble a seed set of high-value customers, using embedding similarity or propensity models. (Ch 10)

**LTV (Lifetime Value)** — The predicted total revenue a customer will generate over a defined future period. Used for audience segmentation, bid optimization, and campaign targeting. (Ch 10)

---

## M

**MAB (Multi-Armed Bandit)** — See [**Bandit Algorithm**](#b).

**Match Type** — The degree of flexibility in keyword matching: exact match (query must match keyword precisely), phrase match (query contains keyword phrase), and broad match (query is semantically related). (Ch 2, Ch 5)

**MRC (Media Rating Council)** — An industry body that sets standards for ad measurement, including the definition of a viewable impression (50% of pixels visible for 1 second for display, 2 seconds for video). (Ch 12)

**Multi-Task Learning** — A training paradigm where a single model simultaneously optimizes for multiple objectives (e.g., CTR and CVR) with shared representations. Improves data efficiency and reduces serving cost. (Ch 6)

**Multi-Touch Attribution (MTA)** — An attribution methodology that allocates conversion credit across multiple ad interactions in a user's path, rather than assigning all credit to a single touchpoint. (Ch 2, Ch 12)

---

## N

**Negative Sampling** — A training technique that generates negative examples (non-clicked or non-converted ad impressions) to train binary classification models. Hard negative mining selects challenging negatives to improve model discrimination. (Ch 6)

---

## O

**OPE (Off-Policy Evaluation)** — A family of methods for estimating the performance of a new policy (e.g., a new ranking model) using data collected under a different policy, without deploying the new policy online. Techniques include Inverse Propensity Scoring (IPS) and Doubly Robust (DR) estimators. (Ch 12)

---

## P

**pCTR (Predicted Click-Through Rate)** — The model's estimated probability that a user will click a given ad in a given context. A core input to auction ranking via the eCPM formula. (Ch 4, Ch 6)

**pCVR (Predicted Conversion Rate)** — The model's estimated probability that a user will convert after seeing or clicking a given ad. Used for ROAS-optimized bidding and value-based ranking. (Ch 4, Ch 6)

**PID Controller** — A feedback control mechanism (Proportional-Integral-Derivative) adapted for budget pacing. Adjusts a bid multiplier based on the error between target and actual spend rate. (Ch 7)

**Placement** — The specific location on a page where ads can appear: search results page (SRP) top slot, product detail page (PDP) sidebar, homepage carousel, etc. Different placements have different engagement rates and may require placement-specific models. (Ch 2, Ch 3, Ch 6)

**PQ (Product Quantization)** — A vector compression technique that splits high-dimensional vectors into sub-vectors and quantizes each independently, enabling memory-efficient ANN search. Often combined with IVF. (Ch 5, Ap 1)

**Propensity Model** — A model that predicts the probability a user will take a specific action (e.g., purchase in a product category within 7 days). Used for audience construction and targeting. (Ch 10)

**Pseudonymized ID** — A hashed or tokenized user identifier that preserves privacy while enabling cross-session targeting and attribution. Cannot be reversed to reveal the original identifier without additional information. (Ch 2, Ch 13)

---

## Q

**QPS (Queries Per Second)** — A throughput metric measuring the number of ad requests a system can handle per second. Production RMN systems typically target 10K to 50K+ QPS. (Ch 3)

**Quality Score** — A composite metric combining predicted engagement (pCTR, pCVR), ad relevance, and landing page quality. Used as a multiplier in auction ranking to reward high-quality ads. (Ch 1, Ch 4)

---

## R

**RampID** — LiveRamp's deterministic identity solution that resolves individuals across devices and channels using an identity graph built from authenticated events. (Ch 13, Ap 3)

**Reserve Price** — The minimum bid required for an ad to participate in an auction. Protects marketplace value and prevents low-quality ads from winning at trivially low prices. (Ch 2, Ch 4)

**Retrieval** — The first stage of the ad-serving pipeline that generates a candidate set of potentially relevant ads from the full index. Combines boolean targeting (inverted index) and semantic similarity (ANN vector search). Also called recall or candidate generation. (Ch 3, Ch 5)

**RMN (Retail Media Network)** — An advertising platform operated by a retailer, using first-party shopper data to serve ads on retailer-owned properties (on-site) or across the open web (off-site). Distinguished from traditional ad networks by deterministic attribution, first-party data, and closed-loop measurement. (Ch 2)

**ROAS (Return on Ad Spend)** — The revenue attributed to a campaign divided by its cost: `attributed_revenue / ad_spend`. The primary efficiency metric for most RMN advertisers. (Ch 2, Ch 4, Ch 12)

---

## S

**ScaNN (Scalable Nearest Neighbors)** — Google's open-source library for efficient vector similarity search, using anisotropic vector quantization for improved recall at high throughput. (Ch 5)

**Scoring** — The second stage of the ad-serving pipeline that ranks retrieved candidates using a neural model (pCTR, pCVR predictions) to compute an auction score. Also called ranking. (Ch 3, Ch 6)

**Second-Price Auction** — See [**GSP**](#g).

**Segment** — A static or dynamic group of users sharing a characteristic (e.g., "frequent buyers," "new visitors"). Used for audience targeting and reporting breakdowns. (Ch 2, Ch 10)

**Shapley Value** — A game-theoretic method for fairly allocating credit among contributors. In attribution, it assigns conversion credit across touchpoints based on their marginal contributions across all possible orderings. (Ch 12)

**SIVT (Sophisticated Invalid Traffic)** — Non-human or fraudulent traffic that requires advanced detection methods (behavioral analysis, ML models, device fingerprinting) to identify. Includes bots mimicking human behavior, click farms, and ad stacking. Contrasted with GIVT. (Ch 11)

**SLA (Service Level Agreement)** — A formal commitment defining system performance requirements, typically including latency percentiles (e.g., p99 < 100ms) and availability targets (e.g., 99.9% uptime). (Ch 3)

**Sponsored Product** — An ad format where a product listing is promoted to a higher position in search results or category pages, appearing alongside organic results. The dominant ad format on RMN platforms. (Ch 2)

---

## T

**Thompson Sampling** — A Bayesian bandit algorithm that selects actions by sampling from the posterior distribution of each action's expected reward. Naturally balances exploration and exploitation. (Ch 9)

**Throttling** — A pacing mechanism that probabilistically drops ad requests to control spend rate. When the budget is being consumed too quickly, the throttle rate increases, reducing the fraction of auctions the campaign enters. (Ch 7)

**Tower** — In multi-tower neural architectures, an independent sub-network that processes one category of input features (e.g., user tower, ad tower, context tower) and produces an embedding vector. Tower outputs are combined in a fusion layer. (Ch 5, Ch 6)

**Two-Tower Model** — A neural architecture with separate encoder towers for queries (users) and items (ads) that produce embeddings in a shared vector space. Enables pre-computation of item embeddings for fast ANN retrieval. (Ch 5, Ch 6)

---

## U

**UCB (Upper Confidence Bound)** — A bandit algorithm that selects the action with the highest upper confidence bound on its estimated reward, balancing the mean estimate with an exploration bonus inversely proportional to the number of times the action has been tried. (Ch 9)

**UID2 (Unified ID 2.0)** — An open-source, interoperable identity framework that creates encrypted, rotating tokens from authenticated user signals (email, phone). Designed as a privacy-preserving replacement for third-party cookies. (Ch 13, Ap 3)

**Uplift Model** — A model that predicts the incremental effect of a treatment (ad exposure) on a user's behavior, rather than predicting the outcome itself. Used for targeting users who are most responsive to advertising. (Ch 10, Ch 12)

---

## V

**VCG (Vickrey-Clarke-Groves) Auction** — A truthful auction mechanism where each winner pays the externality they impose on other bidders. Guarantees incentive compatibility (bidding true value is the dominant strategy). (Ch 4)

**Viewability** — The measure of whether an ad was actually seen by a user, per MRC standards: 50% of pixels in view for at least 1 second (display) or 2 seconds (video). (Ch 12)

---

## W

**Wide & Deep** — A neural architecture combining a wide (linear) component for memorization of feature interactions with a deep component for generalization. Introduced by Google for recommendation and ranking. (Ch 6)
