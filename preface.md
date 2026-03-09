# Preface

## Why This Book

I worked on Google Display Ads from 2010 to 2015, focusing on data engineering and machine learning pipelines for CTR/CVR prediction and bid landscapes. At the time, logistic regression with FTRL-Proximal, inverted-index retrieval, and nightly batch ETL were the state of the art—and they worked remarkably well. Those systems served billions of ads per day with single-digit-millisecond scoring latency, and the lessons from that era—calibration discipline, online learning, careful data pipelines—remain foundational.

Over the past year I revisited how modern ad stacks are implemented, especially the AI/ML components, and the distance traveled was staggering. Deep learning, semantic embeddings, transformer-based architectures, and generative AI have fundamentally reshaped every stage of the pipeline: retrieval, ranking, creative optimization, and measurement.

At the same time, the industry's center of gravity has shifted. Retail media networks (RMNs) have captured most incremental advertising growth versus earlier third-party networks. By 2021, Amazon Ads alone exceeded $30 billion in annual revenue; Walmart Connect, Instacart Ads, and Kroger Precision Marketing scaled rapidly behind it. RMNs co-locate publisher inventory, audiences, landing experiences, and conversions inside the same retail platform. This vertical integration—combined with first-party data ownership and deterministic attribution—creates both a richer engineering canvas and a more demanding set of constraints than traditional ad networks ever faced.

Yet there is no practitioner-oriented reference that bridges the gap between legacy ad-tech architectures and the AI-native systems powering modern RMNs. Academic papers cover individual components (two-tower retrieval, multi-task ranking, dynamic creative optimization) but rarely show how these pieces compose into a production system operating at 50K QPS under 50ms latency budgets. This book fills that gap.

## Who Should Read This Book

This book is written for **senior ML engineers, data scientists, and ad-tech architects** who need to design, build, and operate production-grade retail media advertising platforms. We assume familiarity with machine learning fundamentals (supervised learning, neural networks, embeddings), systems engineering basics (distributed systems, latency/throughput tradeoffs), and at least one deep-learning framework (we use PyTorch throughout).

If you are new to advertising technology, start with Chapters 1 and 2—they provide the domain grounding you need. If you are an experienced ad-tech engineer, you may skim those chapters and jump directly to the systems and modeling content in Part II.

## How the Book Is Organized

The book follows the natural architecture of an RMN ad-serving platform, moving from foundations through core systems to advanced capabilities.

**Part I — Foundations** establishes the conceptual baseline:

- **Chapter 1: Traditional Ad Network Architecture** covers the display and search ad systems of the 2010–2015 era—inverted-index retrieval, logistic regression CTR models, FTRL-Proximal optimization, GSP auctions, CVR modeling, bid landscapes, and batch data pipelines. This is the technical "before" picture that the rest of the book builds upon.

- **Chapter 2: The Retail Media Network Ecosystem** defines what RMNs are, why they emerged, and what makes them architecturally distinct from traditional networks. It maps the domain—campaign hierarchies, targeting types, bidding mechanics, measurement—so that engineering decisions in later chapters are grounded in business context.

**Part II — Core Serving Stack** builds the real-time ad-serving pipeline end to end:

- **Chapter 3: Ad Serving Architecture** designs the service topology, defines SLAs, and traces the full request flow: load balancer → ad server → retrieval → scoring → auction → creative assembly → logging.

- **Chapter 4: Auction and Cost Tracking** implements adjusted eCPM ranking, quality scores, GSP pricing, and the cost-tracking infrastructure that feeds advertiser billing and reporting.

- **Chapter 5: Ads Retrieval** solves the candidate generation problem—how to find relevant ads from millions of candidates in under 15ms using hybrid retrieval (Boolean targeting + semantic vector search via ANN indices).

- **Chapter 6: Multi-Tower Scoring Model** details the neural architecture for CTR/CVR prediction—multi-tower design, fusion layers, multi-task learning, calibration, and serving optimization.

- **Chapter 7: Budget Pacing and Spend Control** addresses real-time budget management—PID controllers, throttling, Lagrangian pacing, and the infrastructure for smooth spend allocation.

**Part III — Media Planning & Optimization** covers the systems that shape what the shopper sees and how audiences are built:

- **Chapter 8: Automated Creative Generation** builds the offline pipeline for LLM-driven copy, diffusion-based image generation, layout assembly, and governance gates.

- **Chapter 9: Creative Selection and Learning** covers online dynamic creative optimization—bandit-based selection, reward modeling, and closed-loop learning from impression outcomes.

- **Chapter 10: Predictive Audiences** constructs ML-driven audience segments—propensity modeling, lifetime value estimation, lookalike expansion, and privacy-safe activation via clean rooms.

**Part IV — Trust, Measurement & Governance** closes the feedback loop and safeguards the platform:

- **Chapter 11: Ad Fraud Detection and Prevention** builds the multi-layer fraud defense system—rule-based filters, behavioral analysis, ML scoring, device fingerprinting, and advertiser refund workflows.

- **Chapter 12: Measurement, Attribution, and Incrementality** engineers the event-joining, attribution, and incrementality infrastructure that closes the feedback loop between ad serving and campaign performance.

- **Chapter 13: Privacy, Governance & Identity Resolution** implements privacy-preserving identity resolution, consent management, differential privacy for aggregate reporting, and compliance monitoring.

**Part V — Appendices** provides deep technical references for practitioners:

- **Appendix 1: AdIndex System Implementation** delivers detailed implementation guidance for building AdIndex from scratch—data models, online query processing, asynchronous updates, sharding, and replication strategies for high availability.

- **Appendix 2: Embedding Architectures for Sequential Features** covers the technical foundations for embedding-based feature encoding used in the multi-tower scoring model—neural embeddings, sequential encoders for text, and multi-dimensional behavioral action embeddings.

- **Appendix 3: Identity Solution Implementations** provides specific technical details for deploying privacy-preserving identity solutions—UID2 reference implementation, RampID, ID5, EUID, and multi-vendor integration strategies.

A **Glossary** in the back matter collects definitions of key terms, acronyms, and concepts from all chapters into a single alphabetical reference.

Each chapter from Part II onward contains production-quality PyTorch code examples, Mermaid architecture diagrams, and concrete latency/scale targets grounded in real-world RMN operating conditions.

**The technology evolution at a glance.** The progression from traditional to modern ad networks spans five dimensions:

| Dimension | Traditional (Ch 1) | Modern RMN (Ch 3–13) |
|-----------|--------------------|-----------------------|
| **Retrieval** | Inverted indices, boolean matching | Two-tower embeddings, ANN vector search (FAISS/ScaNN/HNSW) |
| **Ranking** | Logistic regression + feature crosses | Wide&Deep, DeepFM, DCN, DIN/DIEN, Transformer models |
| **Audiences** | Coarse cookie-based segments | Predictive propensity (purchase-in-7d), LTV, uplift models |
| **Creative** | Manual static assets | LLM/diffusion generation, contextual bandits, DCO |
| **Operations** | Nightly batch ETL | Streaming feature stores, continuous calibration, clean rooms |

## How to Use the Code

All code examples are tested in Jupyter notebooks located in the `code/` directory (e.g., `code/ch6_multi_tower_scoring_model.ipynb`). The notebooks are the source of truth—chapter markdown includes the same code verbatim. We use **PyTorch** as the primary framework, with supporting libraries noted in each chapter.

## Acknowledgments

*(To be completed.)*
