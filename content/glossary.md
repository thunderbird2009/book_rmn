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

**ACoS (Advertising Cost of Sales)** — An advertiser efficiency metric equal to ad spend divided by attributed revenue (`ad_spend / attributed_revenue`), the inverse of ROAS. Target ACoS bidding sets a maximum CPC bid of `Target ACoS × expected revenue × pCVR` and lets shadow pricing shade bids toward that ceiling based on budget utilization. (Ch 7)

**Ad** — The atomic auction candidate and the lowest level of the Campaign → Ad Group → Ad hierarchy: a reference to one SKU for sponsored products, or a creative for display. (Ch 2)

**Ad Campaign Service (AdSvc)** — The service that supplies authoritative bid properties (max bid, bid strategy) and campaign metadata (advertiser ID, budget status, policy flags) for the reduced candidate set entering the auction. (Ch 3)

**Ad Event Service (AdEventSvc)** — The service that receives click and viewability events from the frontend, validates them for fraud and duplication, and forwards pre-calculated auction costs to BudgetSvc for budget deduction. (Ch 4)

**Ad Exchange** — A marketplace where publishers' impressions, usually represented by SSPs, are auctioned in real time to bidders, usually DSPs acting for advertisers. (Ch 1)

**Ad Format** — The kind of ad unit a shopper sees: sponsored product, display ad, brand unit or video. Independent of the **Pricing Model** and the **Buying Model**. (Ch 2)

**Ad Group** — The middle level of the Campaign → Ad Group → Ad hierarchy: a set of ads that share targeting and bid settings. In display, the ad group is called an **Ads Line Item** and also carries delivery terms. For sponsored products, each (SKU, ad group) pair is an **Advertised Product**. (Ch 2, Ch 4)

**Ad Index (AdIndex)** — The service and in-memory data structure that stores and indexes all active ad candidates and runs the retrieval stage of the serving pipeline. It holds multiple index structures (keyword, product text, product ANN, product-to-product, and the ads line item DNF index) behind one request path, returns format-tagged candidates, and completes its work within the 15 ms p99 retrieval budget. (Ch 3, Ch 4, Ap 1)

**Ad Load** — How many ad slots a page carries, and where. Set by the page-arbitration policy that balances ad revenue against organic results and shopper trust. (Ch 2)

**Ad Rank** — In generalized second-price (GSP) auctions, the product of an advertiser's bid and Quality Score: `Ad Rank = bid × Quality Score`. Determines both the ad's position and, via the runner-up's Ad Rank, the minimum price needed to win. (Ch 1)

**Ad Relevance** — The Quality Score component that consumes the relevance model's score. The relevance model is built in Chapter 5 from relevance judgments rather than engagement; Chapter 6 consumes its score inside Quality Score. Distinct from pCTR because it is independent of any single user's click propensity. (Ch 5, Ch 6)

**Ad Server** — The central service that orchestrates the real-time ad request pipeline: receiving a request, calling retrieval and scoring services, running the auction, selecting the winner, assembling the creative, and logging the event. (Ch 3)

**Adjusted eCPM** — The final auction ranking metric combining bid, predicted click-through rate, and Quality Score: `Adjusted eCPM = Bid × pCTR × 1000 × Quality Score`. Extends the base eCPM formula with a quality multiplier. (Ch 3, Ch 4)

**Ads Line Item** — The display form of the ad group. Besides targeting and creatives, it carries the delivery terms of a display buy: placements, flight dates, an impression goal or budget, a price, a priority tier (sponsorship, guaranteed, standard, remnant) and a frequency cap. (Ch 2)

**Adstock** — A transformation applied to marketing spend in media mix models that captures the lagged, carry-over effect of advertising, so a dollar spent this period still contributes to the response in later periods. The geometric form, $\text{adstock}_t = x_t + \lambda \cdot \text{adstock}_{t-1}$, decays the carry-over by a channel-specific rate $\lambda \in [0, 1)$. (Ch 14)

**Advertised Product** — The sponsored-products retrieval unit: an (SKU, ad group) pair. One SKU can be advertised in several ad groups with different targeting, bids and budgets, so each pair is a separate candidate until duplicates are removed from the page. (Ch 1, Ch 2)

**Advertising Token** — UID2/EUID terminology for the short-lived, Base64-encoded ephemeral token (AES-256-GCM encrypted raw UID2 plus metadata) that publishers attach to ad calls and bid requests; distinct from the longer-lived refresh token used only to obtain new advertising tokens. (Ap 3)

**ANN (Approximate Nearest Neighbor)** — A family of algorithms that find approximate (rather than exact) nearest neighbors in high-dimensional vector spaces, trading a small amount of recall for orders-of-magnitude speedup. Key implementations include FAISS, ScaNN, and HNSW. (Ch 4, Ap 1)

**Attribution** — The process of assigning credit for a conversion to one or more prior ad interactions (impressions, clicks). RMNs benefit from deterministic attribution via first-party data, unlike traditional networks that rely on probabilistic methods. (Ch 2, Ch 12)

**Attribution Window** — The time period after an ad interaction during which subsequent conversions are credited to that interaction. Typical windows range from 1 to 30 days. (Ch 2, Ch 12)

**Auction** — The mechanism by which ad candidates compete for an impression. The ad server evaluates bids, predicted engagement, and quality scores to determine the winner and the price paid. (Ch 4)

**Audience Targeting** — Ad eligibility determined by boolean rules over first-party user attributes (behavioral, transactional, or predictive), evaluated against a feature store at serve time. Works on every placement type, unlike keyword or contextual targeting. (Ch 2, Ch 10)

**Auto-Bidding** — An automated bidding strategy where the platform adjusts bids per auction to optimize toward an advertiser-specified goal (e.g., Target ROAS, Target CPA, Maximize Conversions). (Ch 4, Ch 7)

**Auto-Targeting** — Sponsored-products targeting in which the advertiser supplies no keywords and the platform matches the product to queries and product pages from the product itself (for example, close match, loose match, substitutes and complements). See also **Keyword Targeting**. (Ch 2)

**Average Treatment Effect on the Treated (ATT)** — In an incrementality experiment, the causal effect measured specifically on users who were actually exposed to treatment, as opposed to everyone assigned to the treatment group. Contrast with [**Intent-to-Treat (ITT)**](#i), the more conservative and more commonly reported metric. (Ch 14)

---

## B

**Bandit Algorithm** — An online learning algorithm that balances exploration (trying less-known options) with exploitation (choosing the current best option). Used for creative selection, audience exploration, and dynamic optimization. Common variants include Thompson Sampling and Upper Confidence Bound (UCB). (Ch 9)

**Bayesian Optimization** — A sequential model-based optimization technique that builds a probabilistic surrogate model (typically a Gaussian Process) of an expensive objective function and uses it to select the most promising next evaluation point, minimizing the number of evaluations needed relative to grid or random search. Used to tune quality-score weights and other auction hyperparameters against replayed auction logs. (Ch 4)

**Behavioral Biometrics** — Fraud-detection signals derived from the fine-grained motor patterns of human input, such as mouse trajectory curvature, click-offset distribution, scroll jitter, and inter-key timing cadence, which are difficult for automated scripts to replicate convincingly even when higher-level behavior is scripted to mimic a human session. (Ch 11)

**BERT4Rec** — A bidirectional self-attention model for sequential recommendation, trained with a masked-item ("Cloze") objective that predicts a masked event from both its left and right context. The architecture pattern behind transformer-based user sequence encoders used for propensity scoring. (Ch 10)

**Bi-Encoder** — A model with independent encoders for the query and the item, whose output vectors are compared with a dot product. The two-tower model behind embedding-based retrieval. See also **Two-Tower Model**. (Ch 5)

**Bid Landscape** — The market-side view of an auction: the distribution of the price an ad must beat to win, or equivalently its win probability as a function of bid. Under autobidding, where advertisers no longer vary bids manually, RMNs estimate it via survival analysis (Kaplan-Meier, Cox proportional hazards) over right-censored auction outcomes rather than a simple empirical CDF. Historically the term also covered advertiser-side outcome-versus-bid estimates, which this book calls a **Bid Response Curve**. (Ch 1, Ch 4, Ch 14)

**Bid Response Curve** — The advertiser-side view: the expected impressions, clicks, conversions and spend an advertiser receives as a function of its bid. Derived from the market-side **Bid Landscape** combined with the ad's own click and conversion rates; shown to advertisers by bid simulators. (Ch 1)

**Bid Shading** — A market-facing tactical adjustment applied to a submitted bid to account for auction mechanics, most important in first-price auctions where the bid paid equals the bid submitted. Distinct from shadow pricing, which is an internal, auction-agnostic valuation adjustment for budget-constrained optimization. (Ch 7)

**Billing Service (BillingSvc)** — The financial backend that aggregates daily costs from BudgetSvc into advertiser invoices, processes payments, applies tax calculations, and handles refunds and adjustments. (Ch 4)

**BM25** — The standard lexical relevance scoring function, built on the probabilistic relevance framework. It combines inverse document frequency with term-frequency saturation and document-length normalization; the field-weighted BM25F variant scores structured product fields (title, brand, attributes) separately and combines them. (Ch 4)

**Boolean Targeting** — Targeting expressed as Boolean rules (AND, OR, NOT) over keywords, audience segments, categories and context, indexed in disjunctive normal form (see **DNF (Disjunctive Normal Form)**). It is the eligibility layer for sponsored products and the core of display retrieval. (Ch 2, Ch 4, Ap 1)

**Brand Unit** — A hybrid ad unit on search results that combines a brand logo, a headline and several products: targeted like sponsored products (keywords, products) and built like display. (Ch 2)

**Brier Score** — The mean squared error between predicted probabilities and observed binary outcomes, a proper scoring rule that jointly penalizes poor calibration and poor discrimination. Lower values indicate more trustworthy probability estimates. (Ch 10)

**Budget & Pacing Service (BudgetSvc)** — The service that tracks campaign spending in real time, enforces hard budget caps, and applies pacing throttles so campaigns spend evenly rather than exhausting budget early in the day. (Ch 3, Ch 7)

**Budget Pacing** — The process of distributing an advertiser's budget over time (e.g., evenly across a day or flight) to avoid premature exhaustion. Techniques include PID controllers, throttling, and Lagrangian optimization. (Ch 2, Ch 7)

**Buying Model** — How inventory is bought: in an auction, impression by impression, or through **Guaranteed Delivery** and **Sponsorship** deals. Independent of the **Ad Format** and the **Pricing Model**. (Ch 2)

---

## C

**C2PA (Coalition for Content Provenance and Authenticity)** — An industry standard for embedding cryptographically signed provenance metadata ("Content Credentials") in media files, disclosing whether and how an image or video asset was AI-generated or edited. Applied to generated creative assets for regulatory disclosure compliance. (Ch 8)

**Calibration** — The property that a model's predicted probabilities match observed frequencies. A well-calibrated pCTR model predicting 5% click-through rate should see approximately 5 clicks per 100 impressions. Critical for fair auction ranking and accurate billing. Measured per slice (placement, device, position, format) and required across formats so display and sponsored-products scores share units. (Ch 5, Ch 11)

**Campaign** — The top-level advertising entity containing one or more ad groups. Defines overall budget, flight dates, and optimization objectives. (Ch 2)

**Cascade Ranking** — A multi-stage retrieval and ranking pipeline in which progressively more expensive models score progressively smaller candidate sets, so a full model never evaluates the entire corpus. The reference RMN cascade is about 3,000 merged candidates, about 300 after the pre-ranker, about 200 after the relevance gate, and about 30 into the auction. (Ch 4)

**Cannibalization** — Conversions that would have occurred without ad exposure but are attributed to ads. For an RMN the main form is organic cannibalization: a sponsored product capturing a sale the shopper would have made through the organic results, for example when the product would have ranked first organically anyway. Incrementality measurement corrects for it. (Ch 2, Ch 12)

**Causal Forest** — A tree-based method purpose-built for estimating heterogeneous treatment effects, splitting on covariates that best separate units by uplift rather than by outcome, and providing valid confidence intervals on each estimate. (Ch 10)

**CCPA (California Consumer Privacy Act)** — California state law granting consumers rights over their personal data, including the right to know what data is collected, to delete it, and to opt out of its sale. (Ch 13)

**CDC (Change Data Capture)** — A pattern for streaming row-level database changes (inserts, updates, deletes) from a primary database's transaction log to downstream consumers in near real time, typically via a connector such as Debezium publishing to Kafka. Used to propagate bid, budget, and targeting-rule changes from campaign management systems into AdIndex's in-memory structures within the real-time and near-real-time update tiers. (Ap 1)

**Clean Room** — A controlled environment in which two or more parties (e.g., retailer and advertiser) join their data under query and output restrictions, so that each sees only approved, aggregated results and never the other's raw records. RMNs use clean rooms for audience building, activation and measurement with advertisers and media partners. (Ch 1, Ch 2, Ch 10, Ch 13)

**Click Fraud** — Automated or incentivized clicks on an ad that drain advertiser budget without genuine purchase intent, ranging from bot-driven rapid-fire clicking to human click farms and targeted competitor-budget-depletion attacks. Detected through a combination of rule-based rate limits, behavioral session analysis, and ML-based fraud scoring. (Ch 11)

**Click-Through Rate** — See [**CTR**](#c).

**Close Variant** — A query form treated as matching a keyword although not identical to it: misspellings, singular and plural forms, stems, abbreviations and reorderings with the same meaning. (Ch 2)

**Closed-Loop Measurement** — Linking ad exposures to actual sales, online and in store, through the retailer's own customer accounts and loyalty data. (Ch 2)

**Consent Management Platform (CMP)** — A system that collects, stores, and propagates a user's consent choices (e.g., an IAB TCF consent string) to downstream services, gating personalization and data-sharing behavior based on the user's selections. (Ch 13)

**Constrained Decoding** — A generation-time technique that intercepts an LLM's per-token logits before sampling, using a logit processor to blacklist prohibited tokens (set to negative infinity), enforce output-schema structure, or force early termination, rather than filtering invalid output, Ch 4s after generation completes. (Ch 8)

**Contextual Bandit** — A bandit algorithm that conditions its selection policy on contextual features (user attributes, page context) rather than treating each arm identically across all contexts. Used for personalized creative selection. (Ch 9)

**Contextual Targeting** — Ad eligibility determined by the page being viewed (a product, a category, the theme of a query) rather than by user identity. It needs no user features, so it carries no identity exposure, and it can be extended with embedding similarity between ads and pages. (Ch 2, Ch 4)

**Control Plane** — The systems that manage campaigns and serving configuration: advertiser APIs, validation, review, catalog joins, and the propagation path that publishes versioned state to the online fleet. Contrasts with the **Data Plane**, which serves requests from that published state. (Ch 3)

**ControlNet** — A trainable neural network attached alongside a frozen base diffusion model that injects structural conditioning (an edge map, depth map, or segmentation mask) into the decoder at each denoising step, preserving spatial layout and object geometry without modifying the base model's weights. (Ch 8)

**Conversion** — A desired action taken by a user after ad exposure, such as a purchase, add-to-cart, or sign-up. The primary outcome metric for most RMN campaigns. (Ch 2, Ch 12)

**Cost Per Acquisition** — See [**CPA**](#c).

**Cost Per Click** — See [**CPC**](#c).

**Cost Per Mille** — See [**CPM**](#c).

**Cox Proportional Hazards Model** — A semi-parametric survival model in which the hazard rate at time $t$ given covariates $\mathbf{x}$ is the product of an unspecified baseline hazard and an exponentiated linear combination of the covariates, $h(t \mid \mathbf{x}) = h_0(t) \cdot \exp(\boldsymbol{\beta}^\top \mathbf{x})$. Used to condition bid landscape curves on auction context (query category, device, time of day) without assuming a parametric form for the baseline win-rate curve. (Ch 14)

**CPA (Cost Per Acquisition)** — The cost to the advertiser for each conversion: `ad_spend / conversions`. A key efficiency metric for performance campaigns. (Ch 2, Ch 4)

**CPC (Cost Per Click)** — A pricing model where the advertiser pays only when a user clicks the ad. The dominant pricing model in search and sponsored product advertising. (Ch 2, Ch 4)

**CPM (Cost Per Mille)** — A pricing model where the advertiser pays per 1,000 impressions served. Common in display and video advertising. (Ch 2, Ch 4)

**Creative** — The visual or textual asset shown to the user: a product image, banner, video, headline, or body text. May be static (pre-built) or dynamically assembled. (Ch 2, Ch 8, Ch 9)

**Creative Assembly Service (CreativeSvc)** — The service that retrieves creative assets for winning campaigns, applies dynamic substitution (current price, inventory status), enforces placement format compliance, and serializes the final ad payload. (Ch 3)

**Creative Pack** — The stored, versioned bundle of approved creative assets (copy variants, image assets, layout template, provenance metadata) produced by the offline generation pipeline and made available to the online selection service. (Ch 8)

**Cross-Feature Interaction** — A neural network component that explicitly models interactions between features from different towers (e.g., user-ad, user-context). Techniques include cross networks (DCN), factorization machines (DeepFM), and attention mechanisms. (Ch 5, Ap 2)

**Cross-Encoder** — A model that encodes query and item jointly with full cross-attention and reads one score from the joined representation. The most expressive relevance model, but one transformer pass per pair, so it scores few candidates, is cached, or teaches a cheaper student. (Ch 5)

**CTR (Click-Through Rate)** — The ratio of clicks to impressions: `clicks / impressions`. A primary engagement metric used in ranking and billing. (Ch 1, Ch 5)

**CVR (Conversion Rate)** — The ratio of conversions to clicks (post-click CVR) or to impressions (post-view CVR): `conversions / clicks`. Used alongside CTR for multi-task prediction. (Ch 1, Ch 5)

---

## D

**Data Subject Rights** — The set of rights privacy regulations grant individuals over their own data, including the right to access, delete, rectify, port, and object to processing (GDPR Articles 15-21; CCPA/CPRA §1798.100-130). RMNs must automate these as API-driven pipelines rather than manual processes. (Ch 13)

**Data Plane** — The online serving path that answers ad requests from already published state, in contrast to the **Control Plane** that creates and distributes that state. In an RMN it includes request parsing, retrieval, ranking, auction, delivery, and decision logging. (Ch 3)

**DCN (Deep & Cross Network)** — A neural architecture that uses explicit cross layers to model feature interactions alongside a deep network for implicit interactions. Used in ranking models. (Ap 2)

**DCO (Dynamic Creative Optimization)** — The process of automatically assembling and selecting creative elements (headline, image, CTA, layout) in real time based on user context and learned performance. (Ch 8, Ch 9)

**DeepFM** — A neural ranking model combining factorization machines (for second-order feature interactions) with a deep network (for higher-order interactions). (Ap 2)

**Deterministic Attribution** — Attribution that links ad exposure to conversion using the same logged-in user ID, providing high-confidence credit assignment. A key advantage of RMNs over cookie-based networks. (Ch 2, Ch 12)

**Device Fingerprinting** — A probabilistic device identifier constructed from hardware, software, and network signals (canvas/WebGL rendering, TLS handshake parameters, screen and font configuration) that remains stable across sessions even when cookies are cleared or IP addresses rotate. Used to link fraudulent sessions back to a common source device while remaining subject to purpose limitation and data-minimization constraints. (Ch 11)

**Differential Privacy** — A mathematical framework for quantifying and limiting the privacy loss when releasing aggregate statistics computed from individual-level data. Achieved by adding calibrated noise. (Ch 10, Ch 13)

**Display Ad** — An image, video or HTML ad unit in a dedicated slot (homepage hero, category or product-page banner), with creative produced by the advertiser or generated with AI tools. Usually priced by CPM or vCPM and sold by auction or guaranteed deal. Its ad group is the **Ads Line Item**. (Ch 2)

**Distillation** — Training a smaller serving model to imitate a larger teacher. For relevance, a cross-encoder teacher trains a cheaper twin-encoder student, distilling score margins rather than raw scores so thresholds stay meaningful. (Ch 5)

**Double/Debiased Machine Learning (DML)** — A cross-fitting procedure that estimates nuisance components (outcome and treatment-propensity models) on held-out folds before combining them, keeping causal effect estimates such as the conditional average treatment effect valid even when the nuisance components are fit with complex, high-variance ML models. (Ch 10)

**Doubly Robust (DR) Estimation** — An off-policy evaluation technique that combines a direct reward model with inverse-propensity-weighted correction, remaining low-bias and low-variance if either the reward model or the propensity model is accurate, even when the other is not. Used to debias creative performance estimates collected under an exploring bandit policy. (Ch 9)

**DIN (Deep Interest Network)** — A neural architecture that uses a local activation unit to weight historical behaviors by their relevance to the candidate ad. Unlike standard attention, DIN deliberately does not normalize these weights with softmax, so their sum approximates the intensity of the user's activated interest rather than forming a probability distribution. (Ch 5, Ch 11, Ap 3)

**DIEN (Deep Interest Evolution Network)** — An extension of DIN that adds a GRU-based sequential model to capture the temporal evolution of user interests. (Ch 11, Ap 3)

**DNF (Disjunctive Normal Form)** — A boolean expression format (OR of ANDs) used to represent ad targeting rules. Conjunctions carry required (∈) and excluded (∉) attribute values and are partitioned by size for matching. The inverted DNF index is the core of display retrieval; sponsored-products eligibility evaluates the same compiled expressions against each candidate's ad group, looked up by ID. (Ch 4, Ap 1)

**DSP (Demand-Side Platform)** — Software that advertisers and agencies use to buy impressions programmatically across many exchanges and publishers. An RMN acting as a media buyer uses its own or a partner's DSP for offsite media. (Ch 1)

**Dynamic Bidding** — A bid modifier layered on top of an advertiser's manual bid: the platform scales the base bid up or down per auction based on real-time conversion-likelihood signals, while the advertiser retains control of the bid anchor. Distinct from auto-bidding, which removes the per-auction bid entirely. (Ch 2, Ch 4)

---

## E

**ECE (Expected Calibration Error)** — A binned average of the gap between a model's predicted probability and the observed outcome frequency within each bin, used alongside the Brier score to evaluate whether a propensity or CTR/CVR model's probabilities are trustworthy enough to drive threshold-based decisions. (Ch 10)

**eCPM (Effective Cost Per Mille)** — A unified ranking metric that combines bid and predicted engagement to compare ads on different pricing models: `eCPM = bid × pCTR × 1000` (for CPC ads). Enables apples-to-apples comparison across CPC and CPM ads in the same auction. (Ch 2, Ch 4)

**Embedding** — A learned dense vector representation of a discrete entity (user, ad, product, query) in a continuous vector space. Entities with similar characteristics are mapped to nearby points. Embeddings are the foundation of embedding-based retrieval and neural ranking. (Ch 4, Ch 5, Ap 2)

**Embedding-Based Retrieval (EBR)** — Retrieval by vector similarity: a query and a catalog item are encoded into a shared embedding space and the nearest items are returned from an ANN index. Also called semantic or dense retrieval. Covers vocabulary gaps that term matching misses, but has weaker control over exact attributes, model numbers, brands and negation. (Ch 4)

**Endemic (Advertiser)** — A brand that sells its products through the retailer's own catalog, as opposed to a non-endemic advertiser (see **Non-Endemic (Advertiser)**) using the retailer's audience data to promote products sold elsewhere. Endemic brands compete for placements like Sponsored Product slots on their own listings. (Ch 1, Ch 2)

**Envelope Encryption** — LiveRamp's RampID pattern for sharing a resolved identity with specific downstream partners: the identifier is encrypted into a partner-scoped "envelope" that only the named recipient can decrypt, with a configurable TTL (typically 24 hours) and an audit trail of creation and decryption events. (Ap 3)

**Ephemeral Token** — A short-lived, encrypted artifact (e.g., `AES-256-GCM(key, nonce, pseudonym_id || timestamp)`) that carries an identity resolution result across a trust boundary, such as a bid request to an external SSP. Distinct from a pseudonym ID: it is generated fresh per use, rotates with the encryption key, and reveals nothing to a party without the decryption key. (Ch 13)

**ESMM (Entire Space Multi-Task Model)** — A multi-task learning framework that models CVR over the entire impression space (not just clicked impressions) by predicting pCTCVR = pCTR × pCVR as a supervisory signal trained on all impressions, addressing sample selection bias. (Ch 5)

**EUID (European Unified ID)** — The European variant of UID2, adapted for GDPR compliance with stricter consent requirements and EU-based operator infrastructure. (Ch 13, Ap 3)

**Exploration** — In online learning, the deliberate selection of less-certain options (ads, creatives, audiences) to gather information and reduce uncertainty, as opposed to exploitation. (Ch 9)

**Exposure Parity** — A fairness metric for ad delivery: the ratio of `P(shown ad | group A)` to `P(shown ad | group B)` for two demographic groups. A ratio near 1.0 indicates parity; significant deviation (e.g., below 0.8 or above 1.25) warrants investigation for disparate impact. (Ch 13)

---

## F

**FAISS (Facebook AI Similarity Search)** — An open-source library for efficient similarity search and clustering of dense vectors. Supports multiple index types (IVF, PQ, HNSW) for ANN retrieval at scale. (Ch 4, Ap 1)

**Feature Hashing** — A technique (the "hashing trick") that maps categorical feature strings directly to a fixed-size bucket space via a hash function, avoiding the need to maintain explicit vocabularies for high-cardinality categorical and cross features. Enables compact, collision-tolerant sparse models at scale. (Ch 1)

**Feature Store** — A centralized system for storing, managing, and serving pre-computed features for ML models. Provides both batch features (for training) and low-latency online features (for inference). (Ch 3, Ch 5)

**Fill Rate** — The share of ad requests or ad slots that return a paid ad rather than a no-fill or house fallback, measured per placement type. A serving-health metric tracked alongside latency and effective CPM. (Ch 3)

**First-Party Data** — Data collected directly by the retailer from its own customers: search queries, browsing behavior, purchase history, and loyalty information. The foundational data asset for RMNs. (Ch 1, Ch 2, Ch 10)

**Flight** — The scheduled period during which a campaign, ad group or ads line item is active and eligible to serve. (Ch 2)

**Frequency Cap** — The maximum number of exposures per user per period for an ads line item. (Ch 2)

**FTRL-Proximal (Follow-The-Regularized-Leader)** — An online learning algorithm for training linear models with L1 regularization, producing sparse models suitable for high-dimensional feature spaces. Widely used in traditional ad CTR prediction. (Ch 1)

**Fusion Layer** — The neural network component that combines embeddings from multiple towers (user, ad, context) into a joint representation for final prediction. Techniques include concatenation, element-wise product, and attention-based fusion. (Ch 5, Ap 2)

---

## G

**GDPR (General Data Protection Regulation)** — EU regulation governing the collection, processing, and storage of personal data. Requires explicit consent, data minimization, and the right to erasure. (Ch 13)

**Generative Retrieval / Ranking** — Retrieving or ranking by having a sequence model decode item identifiers directly, in the form of semantic IDs, instead of scoring a separately retrieved set. (Ch 5)

**Geo Experiment** — An incrementality-testing design that randomizes treatment and control assignment at the geographic level (DMA, zip code, or region) rather than the user level, mitigating auction-mediated and marketplace interference at the cost of much smaller sample sizes (tens of regions rather than millions of users). Commonly paired with synthetic control methods to construct a counterfactual for the treatment region. (Ch 14)

**GIVT (General Invalid Traffic)** — Non-human traffic that can be identified through routine means such as known data-center IP lists, declared bots, and user-agent filtering. Contrasted with SIVT. (Ch 11)

**GSP (Generalized Second-Price) Auction** — An auction mechanism for multiple ranked slots in which each winner pays the minimum needed to keep its position: in the quality-weighted form, the Ad Rank of the advertiser below divided by its own Quality Score, often plus a minimum increment (e.g., \$0.01). Unlike VCG, it is not truthful. The dominant auction format in search and sponsored-product advertising. (Ch 1, Ch 2, Ch 4)

**Graceful Degradation** — A failure strategy that reduces the amount of work the serving stack performs under pressure, stepping down a prepared ladder (hedges off, fewer candidates, cached scores, pre-ranker order, organic-only) so the user-visible outcome degrades deliberately rather than chaotically. See also **Load Shedding**. (Ch 3)

**Graded Relevance (ESCI)** — A four-grade relevance scale: Exact, Substitute, Complement and Irrelevant, defined in Amazon's ESCI benchmark over about 130,000 queries and 2.6 million judged query-product pairs. (Ch 5)

**Guaranteed Delivery** — A buying model in which the RMN commits to deliver an impression goal for an ads line item within its flight at an agreed price, ahead of auctioned demand; shortfalls are compensated with extra delivery or credit. (Ch 2)

---

## H

**Hard Negative** — A near-miss non-relevant training example, such as an iPhone 14 case for the query "iphone 15 case," used to teach a relevance model fine distinctions. (Ch 5)

**Hashed Identifier** — An email address or phone number that has been normalized and passed through a one-way cryptographic hash (e.g., SHA-256), so two parties can find common customers without exchanging raw contact details. Hashing pseudonymizes rather than anonymizes, so hashed identifiers remain personal data. (Ch 2)

**Hazard Model (Discrete-Time)** — A model that predicts, at each discrete time step, the conditional probability of an event (e.g., purchase) occurring at exactly that step given it has not yet occurred. The cumulative probability of the event by horizon $H$ follows from the product of the per-step survival probabilities, letting one trained model answer any horizon at inference rather than requiring a separate model per horizon. (Ch 10)

**Hedged Request** — A tail-latency mitigation technique in which a client, after its primary request has been outstanding longer than a target percentile's expected latency, sends a duplicate copy of the request to a second, redundant replica and uses whichever response arrives first, discarding the other. Applied to scatter-gather queries against straggling AdIndex shard replicas. (Ch 3, Ap 1)

**Hill Saturation** — A function modeling diminishing marginal returns to spend, $f(x) = x^S / (K^S + x^S)$, where $K$ is the half-saturation point and $S$ controls curve steepness; values of $S > 1$ produce an S-shaped response, while $S \leq 1$ produces returns that diminish from the first dollar. Used in media mix models to map adstocked spend onto a bounded response. (Ch 14)

**HNSW (Hierarchical Navigable Small World)** — A graph-based ANN index that builds a multi-layer proximity graph for fast approximate nearest neighbor search with high recall. (Ch 4, Ap 1)

**Homomorphic Encryption (HE)** — A cryptographic technique that permits computation directly on encrypted data, producing an encrypted result that decrypts to the same value as if the computation had run on the plaintext. Offers strong cryptographic privacy guarantees for clean room queries but at very high computational latency and with a limited set of supported operations. (Ch 13)

**Hot Shard** — A shard that receives disproportionately high query volume relative to its peers, typically because a single popular campaign or targeting criterion is concentrated on it. Mitigated through over-replication (spinning up additional replicas for the affected shard) or a short-TTL read-through cache in front of it. (Ap 1)

**Hybrid Retrieval** — A candidate-generation strategy that runs several retrieval paths (keyword, lexical, embedding-based, product-to-product) and merges their results, rather than relying on a single index. The merged candidates retain per-path scores and ranks for fusion and pre-ranking. (Ch 4)

---

## I

**ID5** — A privacy-focused identity solution that generates partner-specific IDs using probabilistic and deterministic signals without relying on third-party cookies. (Ch 13, Ap 3)

**Identity Graph** — A data structure linking pseudonymized tokens (email hash, device ID, loyalty ID) via typed, consent-scoped edges (e.g., login, loyalty enrollment), enabling cross-device and cross-channel identity resolution without storing or exposing raw PII. (Ch 13)

**Identity Resolution** — Linking records that refer to the same person across systems, either deterministically (a shared identifier such as a logged-in account or hashed email) or probabilistically (statistical matching of weaker signals such as device and location). Needed whenever exposure or conversion happens outside the retailer. (Ch 1)

**Impression** — A single instance of an ad being served. Distinctions include: *served* (ad server returned the ad), *rendered* (displayed on screen), and *viewable* (50%+ visible for 1 second or more per MRC standard). (Ch 2, Ch 12)

**Impression Goal** — The contracted number of (viewable) impressions a guaranteed ads line item must deliver within its flight. (Ch 2)

**Incremental ROAS (iROAS)** — Incremental revenue (revenue caused by the ads, measured against a control group) divided by ad spend. Lower than attributed ROAS whenever ads reach shoppers who would have bought anyway. (Ch 2)

**Incrementality** — The sales or conversions caused by advertising, as opposed to those merely attributed to it: the difference between outcomes with and without the ads, measured with randomized holdouts or geo experiments. For RMNs it is the central credibility question, because sponsored products reach shoppers who are already searching to buy. See also **Cannibalization** and **Incremental ROAS (iROAS)**. (Ch 2, Ch 12, Ch 14)

**Insertion Order (IO)** — The agreement for a display buy, especially a guaranteed one, that specifies the deliverables, their price, the maximum spend and the start and end dates. It covers one or more ads line items. (Ch 2)

**Intent-to-Treat (ITT)** — In an incrementality experiment, the causal effect measured across everyone assigned to the treatment group, regardless of whether they were actually exposed to the ad (due to budget exhaustion or targeting mismatches). The more conservative and more commonly reported metric because it avoids the selection bias of conditioning on exposure. Contrast with [**Average Treatment Effect on the Treated (ATT)**](#a). (Ch 14)

**Inter-Annotator Agreement** — A measure of how consistently independent raters label the same pairs. A label-quality metric used to monitor relevance judges and the policy they follow. (Ch 5)

**Interleaving Test** — An experimentation method for comparing two ranking or selection algorithms that exposes each user to results from both algorithms merged into a single alternating sequence, rather than splitting users into separate A/B groups. Detects preference differences with higher statistical power and smaller sample sizes than a traditional A/B test. (Ch 9)

**Invalid Traffic (IVT)** — The umbrella category for non-human or fraudulent ad traffic, split by the IAB/MRC taxonomy into GIVT (routine, list-filterable) and SIVT (sophisticated, requiring behavioral or model-based detection). IVT rate (blocked plus reversed events divided by total events) is the standard metric for advertiser transparency reporting. (Ch 11)

**Inverted Index** — A data structure mapping terms or attribute values (keywords, product text, audience segments, categories) to the postings of documents or ads that contain them. Enables both keyword and product-text lexical retrieval and boolean-targeting-based eligibility. (Ch 4, Ap 1)

**IP-Adapter** — A lightweight adapter that projects a reference image's CLIP embedding into a diffusion model's cross-attention layers alongside the text embedding, injecting the actual visual appearance of a product or brand asset into generated images without fine-tuning the base model. (Ch 8)

**IPS (Inverse Propensity Scoring)** — An off-policy evaluation estimator that reweights each logged observation by the ratio of the target policy's probability to the logging policy's probability for the action taken, producing an unbiased estimate of a new policy's value from historical data alone. Importance weights are typically clipped to control variance at the cost of some bias. (Ch 12)

**IVF (Inverted File Index)** — A vector quantization technique that partitions the embedding space into clusters (Voronoi cells) and searches only the nearest clusters, reducing ANN search cost. Often combined with PQ. (Ch 4, Ap 1)

---

## K

**k-Anonymity** — A privacy property ensuring that each individual in a released dataset is indistinguishable from at least k-1 other individuals with respect to certain identifying attributes. (Ch 10, Ch 13)

**Kaplan-Meier Estimator** — A non-parametric estimator of a survival function from right-censored data, computed as a product of conditional survival probabilities at each observed event time. Applied to bid landscape construction by treating a lost auction as right-censored at the advertiser's bid and a won auction as an observed event at the clearing price. (Ch 14)

**Keyword Targeting** — Manual targeting in which the advertiser chooses keywords, each with a **Match Type**, plus optional **Negative Keyword**s. Distinct from **Auto-Targeting**, where the platform chooses matching queries from the product itself. (Ch 2, Ch 5)

**Kill Switch** — A single API call that disables a live creative across all contexts and placements within seconds, bypassing the normal selection pipeline; triggered manually by brand-safety reviewers or automatically by post-publication monitoring. (Ch 8)

---

## L

**Lagrangian Pacing** — A budget pacing method that uses Lagrange multipliers to solve the constrained optimization problem of maximizing total value (clicks, conversions) subject to a budget constraint. The multiplier acts as a bid modifier that throttles spend. Also called shadow pricing; see [**Shadow Pricing**](#s). (Ch 7)

**Landing Page Quality** — A component of Quality Score aggregating platform-monitored technical metrics (page load time, mobile-friendliness), behavioral metrics (bounce rate, time on page), and content-quality checks (product availability, price accuracy) into a policy-driven score of the destination page's user experience. Unlike pCVR, it measures observed platform standards rather than predicting an individual user's conversion likelihood. (Ch 4)

**Last-Touch Attribution** — An attribution rule that assigns all conversion credit to the most recent ad interaction before conversion. The default model in most RMNs due to its simplicity, though it ignores earlier awareness touchpoints. (Ch 2, Ch 12)

**Late Fusion** — Combining independently computed tower outputs, and their pairwise dot products, in a fusion head. The design of the multi-tower CTR/CVR ranker, where the ad side is precomputable and the user side is cached. Distinct from **Late Interaction**. (Ch 5)

**Late Interaction** — In information retrieval, a token-level interaction between independently encoded query and document representations, as in ColBERT. Distinct from **Late Fusion**. (Ch 5)

**Lexical Retrieval** — Retrieval by term overlap through an inverted index, scored with a function such as BM25. Strong on exact entities (brands, model numbers) and structured attributes, but blind to synonyms and intent not expressed in the query terms. (Ch 4)

**Lightweight Scoring** — See [**Pre-Ranker**](#p).

**LinUCB** — A contextual bandit algorithm that selects the action maximizing a linear reward estimate plus an upper confidence bound term, $\hat{r}_k + \alpha \sqrt{x_k^\top A_k^{-1} x_k}$, where $x_k$ is the context feature vector and $A_k$ is the ridge-regression precision matrix accumulated from past observations. (Ch 9)

**LLM-as-Judge** — Using an LLM to assign relevance labels at scale from the same rubric a human rater follows, with the model and prompt selected against human gold and audited over time. (Ch 5)

**Load Shedding** — Rejecting excess work before a service saturates so it can focus on the requests it decides to admit; in ad serving, this includes dropping requests whose deadline has already expired. See also **Graceful Degradation**. (Ch 3)

**Lookalike Audience** — An audience segment constructed by finding users who resemble a seed set of high-value customers, using embedding similarity or propensity models. (Ch 10)

**LoRA (Low-Rank Adaptation)** — A parameter-efficient fine-tuning technique that inserts small trainable low-rank matrices ($A$ and $B$, rank $r \ll d$) into a frozen model's weight matrices, so that $W' = W + A \cdot B$ shifts the model's output style (e.g., toward a brand's visual identity) while training only a small fraction of the original parameter count. (Ch 8, Ch 9)

**LTV (Lifetime Value)** — The predicted total revenue a customer will generate over a defined future period. Used for audience segmentation, bid optimization, and campaign targeting. (Ch 10)

---

## M

**MAB (Multi-Armed Bandit)** — See [**Bandit Algorithm**](#b).

**Match Rate** — The share of one party's records that are successfully matched to the other party's identities. It bounds the reach of an activated audience and the coverage of a measurement study. (Ch 2)

**Match Type** — The flexibility of keyword matching: exact (the query equals the keyword, allowing close variants), phrase (the query contains the keyword's words in order) and broad (related queries, in any order, including synonyms). Broad match and close variants are implemented by query rewriting, not literal token matching. See also **Close Variant**. (Ch 2, Ch 5)

**Media Mix Modeling (MMM)** — A regression-based technique that decomposes total sales or conversions into contributions from each marketing channel using aggregate time-series data, typically combining an adstock transformation for carry-over effects with a saturation function for diminishing returns. Modern implementations (Google's Meridian, Meta's Robyn) use Bayesian inference and can be calibrated with incrementality-experiment results. (Ch 14)

**Meta-Learner** — A family of uplift-modeling architectures (including the T-Learner, which differences separate treatment and control models, and the more sophisticated X-Learner and R-Learner) that estimate heterogeneous treatment effects, generally outperforming a simple two-model difference when treatment and control groups are imbalanced in size. (Ch 10)

**ML Inference Service** — A stateless backend that hosts serialized ranking models (TorchScript, SavedModel, ONNX) and returns pCTR/pCVR predictions for a batch of candidates, decoupling ML computation from request orchestration. (Ch 3, Ch 5)

**MRC (Media Rating Council)** — An industry body that sets standards for ad measurement, including the definition of a viewable impression (50% of pixels visible for 1 second for display, 2 seconds for video), and that accredits platforms' invalid-traffic detection capabilities. (Ch 11, Ch 12)

**Multi-Task Learning** — A training paradigm where a single model simultaneously optimizes for multiple objectives (e.g., CTR and CVR) with shared representations. Improves data efficiency and reduces serving cost, at the risk of negative transfer, where one task improves at another's expense. (Ch 5)

**Multi-Touch Attribution (MTA)** — An attribution method that splits conversion credit across the ad interactions in a user's path, by a fixed rule (linear, time decay, position based) or with weights learned from data. It allocates credit among observed touchpoints and does not measure what the ads caused; see **Incrementality**. (Ch 2, Ch 12)

---

## N

**Negative Keyword** — A keyword that excludes an ad from queries that contain it. (Ch 2)

**Negative Sampling** — A training technique that generates negative examples (non-clicked or non-converted ad impressions) to train binary classification models. Hard negative mining selects near-miss negatives, such as the wrong model's accessory, to improve discrimination. (Ch 5)

**New-to-Brand** — An advertiser-facing metric measuring the share of converters making their first purchase from the brand: `first-time_purchasers / total_converters`. Used to evaluate customer-acquisition effectiveness separately from repeat-purchase ROAS. (Ch 2)

**Non-Endemic (Advertiser)** — An advertiser whose products are not sold through the retailer's catalog (e.g., banks, automakers, telecom providers). It buys access to the retailer's audience rather than its shelf, and its conversions usually happen on its own properties. See also **Endemic (Advertiser)**. (Ch 1)

---

## O

**Onsite / Offsite** — Onsite media runs on the retailer's own properties (website, apps, in-store screens). Offsite media runs on third-party inventory (open web, social platforms, connected TV), using the retailer's shopper data for targeting and its sales data for measurement. (Ch 1)

**OPE (Off-Policy Evaluation)** — A family of methods for estimating the performance of a new policy (e.g., a new ranking model) using data collected under a different policy, without deploying the new policy online. Techniques include Inverse Propensity Scoring (IPS) and Doubly Robust (DR) estimators. (Ch 4, Ch 9, Ch 12)

---

## P

**pCTR (Predicted Click-Through Rate)** — The model's estimated probability that a user will click a given ad in a given context. Produced by the ranking stage and a core input to auction ranking via the eCPM formula. (Ch 5, Ch 6)

**pCVR (Predicted Conversion Rate)** — The model's estimated probability that a user will convert after seeing or clicking a given ad. Used for ROAS-optimized bidding and value-based ranking. (Ch 5, Ch 6)

**Persuadables** — In uplift-model user segmentation, the users who would not convert without ad exposure but do convert with it (positive uplift), the only segment where ad spend is genuinely effective. Contrasted with sleeping dogs (negative uplift, ads suppress conversion), lost causes (never convert), and sure things (always convert, so spend is wasted). (Ch 10)

**PID Controller** — A feedback control mechanism (Proportional-Integral-Derivative) adapted for budget pacing. Adjusts a bid multiplier based on the error between target and actual spend rate. (Ch 7)

**Placement** — The specific location on a page where ads can appear: search results page (SRP) top slot, product detail page (PDP) sidebar, homepage carousel, etc. Different placements have different engagement rates and may require placement-specific models. (Ch 2, Ch 3, Ch 5)

**Population Stability Index (PSI)** — A drift metric that compares a current score or feature distribution against a baseline across binned buckets, growing as the two distributions diverge. Commonly thresholded at 0.1 (investigate) and 0.25 (retrain) for propensity-model monitoring. (Ch 10)

**Position Bias / Click Model** — The dependence of observed clicks on rank position rather than relevance alone. Click models correct for it when engagement logs are used as training labels. (Ch 5)

**Positional Embedding** — An additional vector encoding a token's position in a sequence, combined with its token embedding so a permutation-invariant architecture like the Transformer can distinguish order (e.g., "New York to Seattle" from "Seattle to New York"). Fixed variants use deterministic sinusoidal functions; learned variants use a trainable lookup table indexed by position, at the cost of not generalizing beyond the maximum sequence length seen in training. (Ap 3)

**Positive-Unlabeled (PU) Learning** — A semi-supervised classification setting in which only a subset of positive examples is labeled and the remainder of the data, a mixture of unobserved positives and true negatives, is left unlabeled. Fraud detection is a natural fit, since confirmed fraud is rare while most traffic is never definitively confirmed as valid or invalid. (Ch 11)

**Piggyback Retrieval** — Deriving ad candidates from the organic search engine's retrieval set, filtered to advertised products and expanded to ad groups, rather than running a separate ad index. Cheap and aligned with organic relevance, but misses advertised items organic retrieval ranks deep or omits. (Ch 4)

**PQ (Product Quantization)** — A vector compression technique that splits high-dimensional vectors into sub-vectors and quantizes each independently, enabling memory-efficient ANN search. Often combined with IVF. (Ch 4, Ap 1)

**Pre-Ranker** — A lightweight model or scoring function between retrieval and full ranking that cuts the merged candidate set (about 3,000) to a few hundred (about 300) using retrieval evidence, bid and lightweight engagement features, within the index's latency budget. (Ch 4)

**Pricing Model** — How the advertiser is charged: per click (CPC), per thousand impressions (CPM) or per thousand viewable impressions (vCPM). Independent of the **Ad Format** and the **Buying Model**. (Ch 2)

**Privacy Budget** — A per-party allotment of differential-privacy epsilon (ε) that bounds the cumulative privacy loss a clean room or reporting API may impose in a given period (typically per day). Each query consumes epsilon proportional to its sensitivity and noise calibration; once exhausted, further queries are blocked until the budget resets. (Ch 13)

**Privacy by Design** — A systems-engineering framework (Cavoukian, 2009) built on seven principles (proactive not reactive, privacy as the default, privacy embedded in design, full functionality, end-to-end security, visibility and transparency, and user-centric control) for embedding privacy protection into a system's architecture rather than adding it after the fact. (Ch 13)

**Product Catalog Service (ProductSvc)** — The service that supplies product details (title, image, price, inventory) for rendering ad creatives and search results, shared across ad serving and organic search. (Ch 3)

**Product Targeting** — Targeting specific products, categories, brands or product features, for example competitors' product pages (conquest) or complementary products. (Ch 2)

**Product-to-Product Retrieval** — Retrieving ads related to the product a shopper is viewing, either substitutes or complements, from behavioral co-view/co-purchase relations, content-based relations, or advertiser product targets. It serves product-page placements and the substitute and complement groups of auto-targeting. (Ch 2, Ch 4)

**Propensity Model** — A model that predicts the probability a user will take a specific action (e.g., purchase in a product category within 7 days). Used for audience construction and targeting. (Ch 10)

**Propensity Score** — In off-policy evaluation, the probability that a logging (or target) policy assigns to the specific action actually taken, given the context. Distinct from the Propensity Model entry above; read directly from the serving-time selection distribution (e.g., a temperature-scaled softmax) rather than estimated after the fact, it is the denominator (or numerator) of the importance weight used by IPS and Doubly Robust estimators. (Ch 12)

**PSA Control** — A holdout-experiment design in which the control group receives a neutral public-service-style ad instead of no ad at all, isolating the effect of the specific ad message from the mere effect of ad-slot exposure. (Ch 2, Ch 14)

**Pseudonymized ID** — A hashed or tokenized user identifier that preserves privacy while enabling cross-session targeting and attribution. Cannot be reversed to reveal the original identifier without additional information. (Ch 2, Ch 13)

---

## Q

**QPS (Queries Per Second)** — A throughput metric measuring the number of ad requests a system can handle per second. Production RMN systems typically target 10K to 50K+ QPS. (Ch 3)

**Quality Score** — A composite quality multiplier in auction ranking that rewards high-quality ads. In the classic search formulation (`Ad Rank = bid × Quality Score`, Ch 1) it includes expected click-through rate along with ad relevance and landing page quality. In the RMN auction, pCTR is a separate term of Adjusted eCPM, and the Quality Score covers the remaining components (pCVR, ad relevance, landing page quality), so click-through rate is counted once. (Ch 1, Ch 3, Ch 4)

**Query Rewriting** — Transforming or expanding a shopper query before matching: spelling correction, synonyms, close variants, and model-generated rewrites that bridge the vocabulary gap between shopper language and catalog language, especially for long-tail queries. (Ch 4)

**Quorum Write** — A replication acknowledgment strategy in which a write is considered durable once it has been propagated to and acknowledged by a majority (or other configured minimum) of replicas, rather than waiting for every replica or accepting acknowledgment from only the primary. Balances write durability against write latency in a single-primary, multi-replica shard. (Ap 1)

---

## R

**RAG (Retrieval-Augmented Generation)** — A generation technique that retrieves relevant context (few-shot exemplars, style guides, historical top-performing content) from a vector store and injects it into an LLM's prompt, grounding output in retrieved evidence rather than relying solely on the model's parametric knowledge. (Ch 8, Ch 9)

**RampID** — LiveRamp's deterministic identity solution that resolves individuals across devices and channels using an identity graph built from authenticated events. (Ch 13, Ap 3)

**Reach** — The number of unique users exposed to an ad or campaign; with frequency, the main delivery metric for display. (Ch 2)

**Reference RMN** — The book's fictional composite retailer, at the scale of the largest US retail media networks, whose traffic, catalog, latency budgets and cascade sizes all chapters use. (Ch 2)

**Relevance Gate** — A thresholded relevance score that removes off-topic ad candidates regardless of bid, separating the question of whether an ad may serve (eligibility) from whether it belongs to this request (relevance). Runs at merge time for cached head-query scores, or after the pre-ranker for tail queries. (Ch 4)

**Relevance Guidelines** — The written, versioned rating policy that defines the graded relevance scale and its examples for raters and LLM judges. (Ch 5)

**Relevance Model** — A model that predicts whether an ad is on-topic for a query or page, trained on relevance judgments rather than clicks. User-independent and cacheable, it gates the auction and its score feeds the Quality Score. (Ch 5)

**Regret (Bandit Regret)** — The cumulative gap in realized value between a bandit's exploration policy and a hypothetical fully-exploit policy that always chose the best-known action. Tracked per audience segment to flag cases where exploration cost is disproportionate to the information gained, and to detect a miscalibrated exploration strategy. (Ch 9)

**Remnant Inventory** — Impressions not committed to guaranteed display deals, sold instead through the auction. The residual supply that remains after guaranteed line items take their delivery. (Ch 3, Ch 16)

**Replication** — Maintaining multiple copies of a shard's data across separate machines, typically as a single writable primary plus several read-oseveral paths: keyword lookup, lexical search over product text, embedding-based (ANN) search, and product-to-product relations. Also called recall or candidate generation. (Ch 3, Ch 4oughput. Replicas commonly serve slightly stale results relative to the primary, which is acceptable when downstream systems already re-verify freshness-sensitive fields (bid, budget) before acting on them. (Ap 1)

**Reserve Price** — The minimum bid required for an ad to participate in an auction. Protects marketplace value and prevents low-quality ads from winning at trivially low prices. (Ch 2, Ch 4)

**Retrieval** — The first stage of the ad-serving pipeline that generates a candidate set of potentially relevant ads from the full index. Runs several retrieval paths (keyword, lexical, embedding-based, product-to-product) and merges their results. Also called recall or candidate generation. (Ch 3, Ch 4)

**Return-Cycle Fraud** — An RMN-specific fraud pattern with no analogue in traditional display advertising: an ad-attributed purchase is returned after the click and conversion have already been billed to the advertiser, inflating reported ROAS while the underlying economic value is zero or negative. Detected by comparing return rates and timing between ad-attributed and organic purchases. (Ch 11)

**Reward Model** — A model trained to predict the performance of a candidate output before or after it is served, used to guide search, ranking, or fine-tuning toward higher-performing variants. In creative optimization, a two-level architecture pairs a fast prompt-level surrogate (scores a configuration before rendering) with an accurate image-level model (scores the rendered asset), with the latter periodically distilled into the former. (Ch 9)

**RFM (Recency, Frequency, Monetary)** — A customer-analytics feature framework built from how recently, how often, and how much a user has purchased. Enhanced versions add velocity terms (rate of change in frequency and monetary value) to capture trajectory, not just current state. (Ch 10)

**RMN (Retail Media Network)** — An advertising business operated by a retailer, using first-party shopper data to serve ads on retailer-owned properties (onsite) and, increasingly, on third-party inventory (offsite). Distinguished from traditional ad networks by vertical integration, deterministic attribution, and closed-loop measurement. (Ch 1, Ch 2)

**RMN Maturity Model** — This book's four-stage framework for how an RMN grows: (1) sponsored products only; (2) ML ranking and auction; (3) display, offsite and data products; (4) agent-ready commerce media. Each chapter's design-decision section gives guidance by stage. (Ch 1)

**RMN Roles (Publisher / Media Buyer / Data Provider)** — The three ways an RMN creates value: selling its own onsite inventory (publisher), buying offsite inventory on advertisers' behalf (media buyer), and selling audiences, clean-room collaboration and measurement (data provider). (Ch 1)

**ROAS (Return on Ad Spend)** — The revenue attributed to a campaign divided by its cost: `attributed_revenue / ad_spend`. The primary efficiency metric for most RMN advertisers. (Ch 2, Ch 4, Ch 12)

**RoPE (Rotary Position Embedding)** — A relative positional-encoding scheme that rotates each pair of query/key dimensions by an angle proportional to token position, rather than adding a separate positional vector to the token embedding. Encodes absolute position while preserving relative-position relationships under the rotation, and generalizes to sequence lengths beyond those seen in training. Used in LLaMA, Mistral, and other modern LLMs. (Ap 2)

---

## S

**Salt Bucket** — A unit of salt-rotation granularity in a pseudonymization scheme: rather than rotating a single global salt (which would invalidate every pseudonym ID at once), PII is hashed with the salt from one of many buckets (e.g., ~1M), and buckets rotate on a staggered schedule so only a small fraction of pseudonym IDs change on any given day. (Ch 13, Ap 3)

**ScaNN (Scalable Nearest Neighbors)** — Google's open-source library for efficient vector similarity search, using anisotropic vector quantization for improved recall at high throughput. (Ch 4)

**Scatter-Gather** — A distributed query execution pattern in which a gateway broadcasts a single incoming query to every shard in parallel ("scatter"), each shard independently retrieves and ranks its local top candidates, and the gateway merges the per-shard results into one final ranked list ("gather"). Query latency is bounded by the slowest responding shard, making tail-latency mitigation such as hedged requests important at scale. (Ap 1)

**Score Fusion** — Combining candidate lists from retrieval paths whose native scores are incomparable (BM25, cosine similarity, match quality, co-engagement). Options include Reciprocal Rank Fusion (RRF), which sums each list's rank contribution as $1/(k+r(d))$ with $k=60$, a learned convex combination of normalized scores, and fixed per-path quotas. (Ch 4)

**Scoring** — The ranking stage of the serving pipeline: a relevance model and a CTR/CVR model score retrieved candidates, after pre-ranking, to compute the auction score. Also called ranking. (Ch 3, Ch 5)

**Second-Price Auction** — See [**GSP**](#g).

**Secure Multi-Party Computation (MPC)** — A cryptographic protocol allowing multiple parties to jointly compute a function over their combined inputs without any party revealing its own input to the others. Used in clean rooms as an alternative to trusted-hardware or governance-only approaches, at the cost of higher latency and setup complexity. (Ch 13)

**Segment** — A static or dynamic group of users sharing a characteristic (e.g., "frequent buyers," "new visitors"). Used for audience targeting and reporting breakdowns. (Ch 2, Ch 10)

**Self-Attention** — An attention mechanism in which every element of a sequence attends to every other element, computing pairwise relevance scores that are used to build a contextualized representation of each token. Processes all positions in parallel (unlike a recurrent encoder), which is what makes positional embeddings necessary. The core mechanism of the Transformer architecture. (Ap 3)

**Shadow Pricing** — The marginal value of one additional dollar of budget, denoted $\psi$; mathematically the Lagrange multiplier on the budget constraint in the Lagrangian relaxation of the constrained value-maximization bidding problem. Used to shade bids down as budget grows scarce, so the campaign wins only impressions where value exceeds cost. Also called Lagrangian pacing; see [**Lagrangian Pacing**](#l). (Ch 7)

**Shapley Value** — A game-theoretic method for fairly allocating credit among contributors. In attribution, it assigns conversion credit across touchpoints based on their marginal contributions across all possible orderings. (Ch 12)

**Sharding** — Horizontally partitioning a dataset and its serving load across multiple machines, each holding a disjoint subset ("shard") of the corpus, so no single machine needs to hold the full dataset in memory or absorb the full query volume. Common shard keys for AdIndex include ad ID (for uniform load distribution) and campaign ID (for co-located campaign-level operations), often combined hierarchically with a geographic first-level split to bound the fan-out of scatter-gather queries. (Ap 1)

**SIVT (Sophisticated Invalid Traffic)** — Non-human or fraudulent traffic that requires advanced detection methods (behavioral analysis, ML models, device fingerprinting) to identify. Includes bots mimicking human behavior, click farms, and ad stacking. Contrasted with GIVT. (Ch 11)

**SKU (Stock-Keeping Unit)** — One distinct item in the retailer's catalog. (Ch 2)

**SLA (Service Level Agreement)** — A formal commitment defining system performance requirements, typically including latency percentiles (e.g., p99 < 100ms) and availability targets (e.g., 99.9% uptime). (Ch 3)

**SNIPS (Self-Normalized Inverse Propensity Scoring)** — A variant of inverse propensity scoring that normalizes the IPS weights to sum to one across the sample, trading a small amount of bias for a substantial reduction in variance relative to plain IPS, particularly when logged propensities are small. (Ch 9)

**Sponsored Product** — An ad format that promotes a specific catalog SKU, rendered natively among organic search results and in product-page and cart carousels, with creative assembled from catalog data. Usually priced per click and sold by auction. Its retrieval unit is the **Advertised Product**. (Ch 2)

**Sponsorship** — A guaranteed, often exclusive buy of a placement for a period, such as a homepage takeover. (Ch 2)

**SSP (Supply-Side Platform)** — Software that publishers use to sell their impressions programmatically through ad exchanges to DSPs. An RMN selling its own onsite inventory occupies the publisher position, with its own ad server doing the selling an SSP would do. (Ch 1)

**Synthetic Control Method** — A causal-inference technique that constructs a counterfactual for a treated unit (e.g., a geo experiment's treatment region) as a weighted combination of untreated control units, with weights chosen to match the treated unit's pre-treatment outcome trajectory. Used to analyze geo experiments with too few regions for conventional statistical inference. (Ch 14)

---

## T

**TCF (Transparency and Consent Framework)** — An IAB Europe industry standard for capturing, storing, and propagating a user's consent and legitimate-interest signals across publishers, vendors, and consent management platforms in programmatic advertising, encoded as a compact TC string. (Ch 13)

**Thompson Sampling** — A Bayesian bandit algorithm that selects actions by sampling from the posterior distribution of each action's expected reward. Naturally balances exploration and exploitation. (Ch 9, Ch 10)

**Throttling** — A pacing mechanism that probabilistically drops ad requests to control spend rate. When the budget is being consumed too quickly, the throttle rate increases, reducing the fraction of auctions the campaign enters. (Ch 7)

**Thundering Herd** — A distributed-systems failure pattern in which many servers act on stale, independently-cached state at once, producing an outcome none of them individually intended. In budget pacing, dozens of servers reading the same cached "budget remaining" value can each approve a bid, causing a large aggregate overspend; the same pattern recurs when a coordinator recovers from an outage and every server simultaneously requests a lease renewal. (Ch 7)

**Time2Vec** — A learned time representation that concatenates a linear component (overall progression) with learned periodic components (cycles at frequencies discovered from data), used as a model-agnostic replacement for hand-crafted timestamp features such as cyclical hour-of-day encoding. (Ch 10)

**TLS Fingerprint (JA3/JA4)** — A hash computed from the parameters of a client's TLS handshake (cipher suites, extensions, elliptic curves) that identifies the underlying TLS client library, distinguishing genuine browsers from headless automation tools even when other signals are spoofed. (Ch 11)

**Tower** — In multi-tower neural architectures, an independent sub-network that processes one category of input features (e.g., user tower, ad tower, context tower) and produces an embedding vector. Tower outputs are combined in a fusion layer. (Ch 4, Ch 5)

**Train/Serve Skew** — A mismatch between how a feature is computed during model training and how the same feature is computed at serving time, caused by duplicated pipeline logic or by using data that would not actually have been available at the historical serving moment. Degrades model accuracy silently, since offline metrics still look correct. Mitigated by a shared feature store and by logging serving-time feature values for reuse in training. (Ch 5)

**Transformer** — A neural architecture built from stacked self-attention and feed-forward layers, processing every element of a sequence in parallel rather than recurrently. The standard architecture for encoding text and behavioral sequences into fixed-size embeddings in modern ad ranking systems. (Ch 5, Ap 3)

**Trusted Execution Environment (TEE)** — A hardware-isolated compute enclave (e.g., Intel SGX, AWS Nitro Enclaves) that runs code and processes data in a protected memory region inaccessible to the host operating system, used by clean rooms to run standard SQL-like queries on combined data at low latency. (Ch 13)

**Two-Tier Budget Architecture** — A distributed budget-management pattern that separates the source of truth (a global coordinator holding authoritative remaining budget) from the decision point (local agents colocated with in embedding-based retrieval. (Ch 4, Ch 5er that make zero-latency bid/no-bid decisions against a small local lease). Bounds worst-case overspend to `lease size × active server count` while keeping the serving path off the network round trip to the coordinator. (Ch 7)

**Two-Tower Model** — A neural architecture with separate encoder towers for queries (users) and items (ads) that produce embeddings in a shared vector space. Enables pre-computation of item embeddings for fast ANN retrieval. The model behind embedding-based retrieval (see **Embedding-Based Retrieval (EBR)**). (Ch 4, Ch 5)

---

## U

**UCB (Upper Confidence Bound)** — A bandit algorithm that selects the action with the highest upper confidence bound on its estimated reward, balancing the mean estimate with an exploration bonus inversely proportional to the number of times the action has been tried. (Ch 9)

**UID2 (Unified ID 2.0)** — An open-source, interoperable identity framework that creates encrypted, rotating tokens from authenticated user signals (email, phone). Designed as a privacy-preserving replacement for third-party cookies. (Ch 13, Ap 3)

**Uplift Model** — A model that predicts the incremental effect of a treatment (ad exposure) on a user's behavior, rather than predicting the outcome itself. Used for targeting users who are most responsive to advertising. (Ch 10, Ch 12)

**User Foundation Model** — A transformer pretrained on long sequences of a shopper's actions across surfaces, producing representations reused by retrieval, ranking, audiences and display targeting. (Ch 5)

---

## V

**VCG (Vickrey-Clarke-Groves) Auction** — A truthful auction mechanism where each winner pays the externality they impose on other bidders. Guarantees incentive compatibility (bidding true value is the dominant strategy). (Ch 4)

**vCPM (Viewable CPM)** — A CPM pricing model that charges advertisers only for impressions that met a viewability threshold (per IAB/MRC standards), rather than for every ad served regardless of whether a user could see it. (Ch 4)

**Vertical Integration** — The RMN structural property of a single retailer owning both the advertising supply (owned properties) and the demand-side data (first-party shopper behavior), in contrast to traditional ad networks that aggregate inventory from many publishers. The source of deterministic attribution and closed-loop measurement. (Ch 1, Ch 2)

**Viewability** — The measure of whether an ad was actually seen by a user, per MRC standards: 50% of pixels in view for at least 1 second (display) or 2 seconds (video). (Ch 4, Ch 11, Ch 12)

---

## W

**Walled Garden** — A large platform (search engine, social network, marketplace) that sells its own inventory with its own first-party data and measurement, largely outside the open programmatic chain. (Ch 1)

**Weak Supervision** — Training labels from noisy heuristic sources, such as catalog rules over category, brand and attribute matches, combined statistically without ground truth. (Ch 5)

**Wide & Deep** — A neural architecture combining a wide (linear) component for memorization of feature interactions with a deep component for generalization. Introduced by Google for recommendation and ranking. (Ap 2)

**Word2Vec** — An early neural word-embedding technique that learns dense vector representations of words from co-occurrence statistics in large text corpora, without labeled supervision. Foundational to later token-embedding approaches used throughout ad ranking systems. (Ap 2)
