# Preface

## Why This Book

Between 2010 and 2015 I worked on Google Display Ads, where my focus was the data engineering and machine learning pipelines behind CTR/CVR prediction and bid landscapes. At the time, the state of the art was logistic regression with FTRL-Proximal, inverted-index retrieval and nightly batch ETL, and those tools worked remarkably well. Those systems served billions of ads a day at single-digit-millisecond scoring latency, and what that era taught (calibration discipline, online learning, scalable data pipelines) still anchors the field.

Today I serve as an Associate Fellow at Walmart Global Tech, where I focus on Walmart Connect, Walmart's retail media network. Over the past year I revisited how modern ad stacks are implemented, especially the AI/ML components, and the distance the field has traveled is staggering. Deep learning, embeddings, transformer-based architectures and generative AI have reshaped every stage of the pipeline, from retrieval and ranking through creative optimization and measurement.

The industry's center of gravity has, at the same time, shifted toward retail media. Amazon's advertising business alone reached \$31.2 billion in revenue in 2021 [1], the first year Amazon reported it separately [2]. In fiscal 2026, Walmart's global advertising business grew 46% to nearly \$6.4 billion, including VIZIO [3], and networks such as Instacart Ads and Kroger Precision Marketing compete alongside them. Across the US market, commerce media advertising grew 18% in 2025 to \$63.4 billion, outpacing internet advertising as a whole (13.9%) and reaching 21.5% of the total [4]. Retail media networks (RMNs) co-locate publisher inventory, audiences, landing experiences and conversions inside the same retail platform. Combined with first-party data ownership and deterministic attribution, this vertical integration creates a richer engineering canvas and a more demanding set of constraints than traditional ad networks ever faced.

Retail media also introduces problems that no traditional network ever had to solve at once. Because the retailer runs the search engine and the ad business on the same pages, every page arbitrates between ad revenue, organic sales and long-term shopper trust. Shoppers on a retail site often arrive already intent on buying, so attributed returns overstate what the ads actually caused, and that gap makes incrementality the central credibility question. A mature RMN plays three roles at once, publisher of its own inventory, buyer of media on other sites and provider of data and measurement; each role that crosses the retailer's boundary brings identity and privacy constraints with it. Shopping agents that browse and buy on a shopper's behalf are now beginning to challenge the assumption that a human sees the page at all.

Yet there is no practitioner-oriented reference that bridges the gap between legacy ad-tech architectures and the AI-native systems powering modern RMNs. Academic papers cover individual components (two-tower retrieval, multi-task ranking, dynamic creative optimization) but rarely show how these pieces compose into a production system that serves 50,000 ad requests per second at peak within 60-millisecond ad-serving budgets on search pages. This book fills that gap architecture first. Each chapter explains how the systems work, names the design decisions they force and states when to choose each option.

## Who Should Read This Book

The primary audience is **senior ML engineers, data scientists, and ad-tech and platform architects** who design, build and operate production retail media platforms. We assume familiarity with machine learning fundamentals (supervised learning, neural networks, embeddings), systems engineering basics (distributed systems, latency and throughput trade-offs) and at least one deep-learning framework; where code appears, it uses PyTorch.

The book also serves **measurement scientists** and **technical product and strategy leads** who need to see how these systems shape what an RMN can promise advertisers and shoppers. Suggested reading paths for each of these groups follow the chapter overview below.

## How the Book Is Organized

The book traces an RMN from its foundations, through the serving path that answers each ad request, and onward to the systems that improve, measure and extend it. Six frameworks recur throughout the book: the RMN maturity model, sponsored products versus display, the ranking cascade, the three roles and the retailer boundary, attribution versus incrementality, and ads versus organic results. Chapter 1, Section 6 explains each one, and later chapters refer to them by name.

**Part I: Foundations** establishes why retail media differs and what its systems must do:

- **Chapter 1: Why Retail Media: From Traditional Ad Networks to RMNs** distills the lessons of the display and search networks that matured by the mid-2010s, explains why their architectures break at retail scale and introduces the RMN's market structure, roles, maturity stages and the book's frameworks.
- **Chapter 2: The RMN Domain and System Map** defines the domain model (campaigns, ad groups, ads line items, advertised products), contrasts sponsored products with display, maps the end-to-end system that the rest of the book builds out and introduces the reference RMN used throughout.

**Part II: The Serving Stack** follows a single ad request from arrival to the page the shopper sees:

- **Chapter 3: Ad Serving Architecture** walks an ad request through the serving path for search and display placements, and shows how the two differ, since display requests carry no query, are driven by the slot and render asynchronously. It describes each component on that path, the latency budget they share and the feedback loops that turn impressions, clicks and conversions into model and pacing updates. It also covers the control plane that carries an advertiser's campaign change from the self-serve console to live serving state.
- **Chapter 4: Ads Retrieval** treats retrieval as two different problems. For sponsored products, it is a relevance search over millions of advertised products, run as parallel paths (lexical and embedding-based retrieval from the shopper's query, product-to-product retrieval from the item being viewed) whose results are merged, gated for relevance, filtered for eligibility and pre-ranked within a 15-millisecond budget. For display, it is an eligibility check over a few thousand ads line items, where a Boolean targeting index does the work; the chapter closes with how one AdIndex service hosts both, and whether ad retrieval should reuse organic search or run its own index.
- **Chapter 5: Ranking Models: Relevance and CTR/CVR Prediction** argues that ranking predicts two different things: whether an ad is on topic, learned from relevance judgments, and whether the shopper will click or buy, learned from logs. It compares the model families used at each stage of the ranking cascade (two-tower retrieval models, multi-tower CTR/CVR models and cross-encoders for relevance), explains why relevance should remain a separate model and shows how relevance labels are produced at scale, from trained raters to LLM judges. It ends with user action sequence foundation models as an emerging shared backbone for retrieval, ranking and audiences.
- **Chapter 6: Auction and Cost Tracking** prices a single auction: adjusted eCPM ranking, Quality Score and how its weights are tuned, generalized second-price and multi-slot pricing, and reserve prices. It weighs ranking on attributed value against incremental value, since sponsoring a product that already ranks first organically earns credit without causing sales. It then follows each cost event through tracking and billing to its hand-off to budget management.
- **Chapter 7: Budget Pacing and Spend Control** keeps roughly sixty serving replicas from overspending a shared budget, using a two-tier design of global coordinators and local budget leases. It develops pacing by throttling participation (probabilistic throttling and feedback control), paces guaranteed display deliveries toward their impression goals and analyzes the failures that cause overspend and underspend. A closing bridge explains how this throttling relates to the bid-scaling pacing of Chapter 9.
- **Chapter 8: Ads and Organic: Page Arbitration and Ad Load** addresses the defining tension of retail media: the retailer's own search results and its ads compete for the same page. It works through the economics of a slot, arbitration policies for search and browse grids (from fixed ad slots to unified ranking under ad-load constraints), module-level allocation on item, home and cart pages, and the blender component that carries out these decisions. It closes with how ad load is set, measured over the long term and governed between the ads and search teams.

**Part III: Optimizing Advertiser Outcomes** covers the three levers an advertiser controls:

- **Chapter 9: Bidding and Autobidding** starts from the market a bidder faces, estimated as bid landscapes and bid response curves, and builds an autobidder for each bid strategy as budget- and ROI-constrained optimization that adapts to delayed, noisy feedback. It covers bidding on incremental rather than attributed value, and how autobidders change the auction itself. It closes with the shift from automating bids to automating whole campaigns.
- **Chapter 10: Creative Optimization: Generation, Selection and Learning** treats creative as one closed loop. Creatives are generated with language and diffusion models or uploaded by advertisers, reviewed against policy and brand-safety rules, selected per impression with bandits and improved from outcomes. The chapter covers sponsored products, display and brand units, and how to detect and refresh creatives that have worn out.
- **Chapter 11: Predictive Audiences** builds audiences from first-party shopping data: propensity, lookalike, lifetime-value and uplift models, together with the sequence-aware features they depend on. It shows how these audiences feed display targeting and offsite activation, and addresses fairness and bias in who gets targeted.

**Part IV: Measurement, Trust and Data** establishes what is real, what was caused and what may leave the retailer:

- **Chapter 12: Ad Fraud Detection and Prevention** starts from the system: a real-time scoring architecture that must decide within the serving latency budget, with failover when it cannot. Detection runs as layers within that architecture, covering click fraud, bots, impression and viewability fraud and conversion fraud, including how to tell legitimate shopping agents from bots. The chapter ends with invalid-traffic classification and advertiser refunds.
- **Chapter 13: Measurement and Attribution** covers measurement inside the retailer, where logged-in shoppers make joins between exposures and purchases deterministic. It engineers the event graph, last-touch and multi-touch attribution, streaming and batch pipelines that reconcile late and offline conversions, offline evaluation of ranking policies and advertiser reporting.
- **Chapter 14: Incrementality and Experimentation** is the book's home for causal measurement. It builds the experimentation platform that internal tests, advertiser campaign experiments and lift studies share; measures lift with holdouts, ghost ads and geo experiments; predicts lift for campaigns that never ran an experiment; and reconciles attribution with incrementality alongside media mix modeling.
- **Chapter 15: Privacy, Identity and Data Collaboration** builds the components that every use of retailer data beyond its own walls depends on: identity resolution, clean rooms, audience activation in both directions and measurement across the retailer boundary. It treats privacy as an architectural constraint rather than a compliance checklist, and covers consent, including consent delegated to shopping agents.

**Part V: Expanding the Network** extends the RMN beyond sponsored products on its own site:

- **Chapter 16: Display Business: Guaranteed Delivery, Reach and Non-Endemic Demand** covers what is unique to selling display: forecasting inventory before it exists, allocating guaranteed deliveries and making good on shortfalls, and managing frequency, reach and brand safety. It also opens the network to non-endemic advertisers, whose conversions happen on their own sites rather than the retailer's.
- **Chapter 17: Offsite Media and the RMN as Data Provider** extends the RMN beyond its own pages. As a media buyer, it reaches shoppers on other sites and on connected TV through DSPs and partners; as a data provider, it sells audiences, clean-room collaboration and closed-loop measurement. The chapter covers the operating models, economics and governance of both roles.

**Part VI: The Road Ahead** looks at what is still unsettled:

- **Chapter 18: Retail Media in the Age of Shopping Agents: Open Problems and Design Directions** examines what happens when agents browse and buy on a shopper's behalf. It maps the problems this creates, labels each current approach as proven, emerging or open, proposes candidate designs and identifies the investments worth making however the market settles.

The **Appendices** provide one reference implementation per topic:

- **Appendix 1: AdIndex System Implementation** details the index structures, query processing, updates, sharding and replication behind Chapter 4.
- **Appendix 2: Reference Ranking Model: Multi-Tower CTR/CVR** implements the multi-tower model with late fusion and multi-task heads behind Chapter 5.
- **Appendix 3: Sequence Models for Users and Queries** covers the encoders for queries and behavioral sequences and the technical details of user foundation models.
- **Appendix 4: Identity Integration Patterns** compares identity solutions and works through one integration end to end.

A **Glossary** in the back matter collects definitions of key terms, acronyms and concepts from all chapters into a single alphabetical reference.

**Reading paths**  
Readers need not go cover to cover. Every path starts with Chapters 1 and 2, which introduce the frameworks and the domain model the rest of the book relies on, and then follows the chapters closest to the reader's work:

- **ML engineers:** Chapters 4, 5, 9, 11 and 14, then Appendices 2 and 3, which follow the models from retrieval and ranking through bidding and audiences to the experiments that evaluate them.
- **Platform architects:** Chapters 3, 4, 6, 7, 8 and 15, then Appendix 1, which follow the serving systems: the request path, the AdIndex service, auction and billing, budget state, the blender, and identity and clean rooms.
- **Measurement and data scientists:** Chapters 13, 14 and 15, then Chapter 11 for uplift-based audiences, which move from attribution inside the retailer to causal measurement and then across the retailer boundary.
- **Product and strategy leads:** Chapters 8, 14, 16, 17 and 18, which cover the decisions that set an RMN's economics and credibility, from ad load to display, offsite and data businesses and shopping agents.

The chapters share a common shape. Each one explains its mechanisms with architecture diagrams, calls out design choices as they arise and closes with a Design Decisions and Trade-offs section that summarizes when to choose each option. The quantitative examples all come from the reference RMN introduced in Chapter 2, a fictional retailer at the scale of a large US retail media network, so traffic, catalog sizes and latency budgets agree across chapters.

## How to Use the Code

Code appears where it clarifies a concept, not as a substitute for explanation. Every executable listing has been tested in a companion Jupyter notebook in the `code/` directory, and appears in the text either whole or as an exact excerpt of a tested cell; the notebooks are the source of truth. A few listings, such as request schemas, interface stubs and calls to vendor APIs that cannot run offline, are marked as illustrative. Complete implementations live in the notebooks and in the reference appendices, such as the multi-tower ranking model in Appendix 2. The examples use PyTorch as the primary framework, with supporting libraries noted where they appear. The notebooks are available in the book's companion repository (https://github.com/thunderbird2009/book_rmn).

## Acknowledgments

*(To be completed.)*

## References

[1] Amazon.com, Inc. (2022). "Amazon.com Announces Fourth Quarter Results." Press release, February 3, 2022. https://ir.aboutamazon.com/news-release/news-release-details/2022/Amazon.com-Announces-Fourth-Quarter-Results/default.aspx (Accessed October 2026). The supplemental net-sales table reports advertising services of \$6,381M, \$7,451M, \$7,612M and \$9,716M for the four quarters of 2021, \$31.16 billion for the year.

[2] Sullivan, L. (2022). "Amazon Ad Business Reaches \$31B, Solidifies Position In Digital's Big 3." *MediaPost*, February 4, 2022. https://www.mediapost.com/publications/article/370906/amazon-ad-business-reaches-31b-solidifies-positi.html (Accessed October 2026). Reports that Amazon disclosed its full-year advertising revenue for the first time with its fourth-quarter 2021 earnings.

[3] Walmart Inc. (2026). "Walmart reports Q4 results." Earnings release, February 19, 2026. https://stock.walmart.com/sec-filings/all-sec-filings/content/0000104169-26-000032/earningsreleasefy26q4.htm (Accessed October 2026). States that Walmart's "global advertising business grew 46% to nearly \$6.4 billion, including VIZIO" in fiscal 2026, and that Walmart Connect in the U.S. grew 41% in the fourth quarter.

[4] Sullivan, L. (2026). "IAB: Ad Revenue Reached \$294.6B In 2025." *MediaPost*, April 16, 2026. https://www.mediapost.com/publications/article/414357/iab-ad-revenue-reached-2946b-in-2025.html (Accessed October 2026). Reports the IAB/PwC Internet Advertising Revenue Report: Full Year 2025 (https://www.iab.com/insights/internet-advertising-revenue-report-full-year-2025/): US internet advertising revenue of \$294.6 billion in 2025 (up 13.9%), with commerce media ad revenue up 18% to \$63.4 billion, 21.5% of media spend.
