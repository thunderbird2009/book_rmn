# Chapter 2: The RMN Domain and System Map

Chapter 1 established why retail media becomes its own engineering problem. This chapter turns that argument into an operating model: the objects the platform stores, the systems that move those objects into serving, and the measurements that come back out. Later chapters assume these definitions are stable. When we say *campaign*, *ad group*, *ads line item*, *advertised product*, *target ROAS* or *incrementality*, this chapter fixes what those terms mean.

It also introduces the **reference RMN**, the fictional large retailer whose traffic, catalog, latency budgets and cascade sizes anchor the rest of the book. That shared profile keeps later numbers honest: a retrieval budget has to fit inside the serving budget, and a candidate count has to match the traffic the platform is built to handle.

Section 1 defines the ontology, the objects the platform stores. Section 2 frames the RMN as a control plane, a serving path and a measurement loop, then anchors later numbers in the reference RMN. Section 3 follows the advertiser through the operating loop. Sections 4 and 5 turn that loop into campaign setup and in-flight controls, and Section 6 closes with the measurement model. Sponsored products remain the primary example, with display defined alongside them.

---
<details>
<summary>Table of Contents</summary>

- [1. Core RMN Ontology](#1-core-rmn-ontology)
  - [1.1. Entity Hierarchy](#11-entity-hierarchy)
  - [1.2. Glossary of Core Terms](#12-glossary-of-core-terms)
- [2. The System Map and a Reference RMN](#2-the-system-map-and-a-reference-rmn)
  - [2.1. The End-to-End System Map](#21-the-end-to-end-system-map)
  - [2.2. A Reference RMN](#22-a-reference-rmn)
- [3. The Advertiser Lifecycle](#3-the-advertiser-lifecycle)
- [4. Ads Planning: Campaign Setup](#4-ads-planning-campaign-setup)
  - [4.1. Campaign Management](#41-campaign-management)
  - [4.2. Ad Formats: Sponsored Products vs. Display](#42-ad-formats-sponsored-products-vs-display)
  - [4.3. Targeting and Audience Strategy](#43-targeting-and-audience-strategy)
    - [4.3.1. Placement Types](#431-placement-types)
    - [4.3.2. Targeting Mechanisms](#432-targeting-mechanisms)
    - [4.3.3. Common Combinations](#433-common-combinations)
- [5. Running Campaigns](#5-running-campaigns)
  - [5.1. Budget Management](#51-budget-management)
  - [5.2. Bidding and Auctions](#52-bidding-and-auctions)
  - [5.3. Advertiser Controls and In-Flight Changes](#53-advertiser-controls-and-in-flight-changes)
- [6. Measurement and Attribution](#6-measurement-and-attribution)
  - [6.1. Core Metrics](#61-core-metrics)
  - [6.2. Attribution Models](#62-attribution-models)
  - [6.3. Incrementality and Experimentation](#63-incrementality-and-experimentation)
  - [6.4. Closing the Loop](#64-closing-the-loop)
- [7. Design Decisions and Trade-offs](#7-design-decisions-and-trade-offs)
- [Summary and Next Steps](#summary-and-next-steps)
- [References and Further Reading](#references-and-further-reading)

</details>

---

## 1. Core RMN Ontology

### 1.1. Entity Hierarchy

The book uses one three-level hierarchy for both ad formats, **Campaign → Ad Group → Ad**, the same hierarchy search networks used (Chapter 1, Section 1.1). Each level has the same job in both formats, but what it contains differs (**Table 2.1**).

**Table 2.1: The Entity Hierarchy by Format**

| Level | Sponsored products | Display |
|---|---|---|
| **Campaign** | Objective, budget, flight, default bid strategy | The same fields for real-time auction. And an **insertion order** for guaranteed and managed buys |
| **Ad group** | Ad group: targeting (keywords, auto-targeting or products), bids, optional budget | **Ads line item**: targeting, placements, flight, impression goal or budget, price, priority tier, frequency cap |
| **Ad** | A reference to one SKU; creative assembled from catalog data | A creative: image, video or HTML asset |
| **Unit retrieved** | **Advertised product**: an (SKU, ad group) pair | An ads line item, then one of its creatives |

A **campaign** is the top-level container, and its **flight** is its active period; ad groups and ads line items have flights too. An **ad group** holds the targeting and bids for a set of ads. An **ad** is the atomic auction candidate: a reference to a **SKU** (stock-keeping unit, one catalog item) for sponsored products, or a creative for display.

**Display: ads line items and insertion orders**  
In display, the ad group is called an **ads line item**, because it carries not only targeting but also the delivery rules the ad server must enforce online: placements, flight, an **impression goal** or a budget, a price, a **priority tier** (sponsorship, guaranteed, auction or remnant) and a frequency cap. The book places guaranteed and managed buys under an **insertion order (IO)**, the sales contract, as most display ad servers do. Auction line items sit directly under the campaign, with no IO between them. Serving works with the ads line item, because that is where the impression goal, timing, priority and pacing rules live. Google Ad Manager's guaranteed line-item types illustrate the split: sponsorship line items deliver a defined share of impressions over a start and end time, while standard line items deliver against an impression goal over a start and end time [2]. Ad servers name these levels differently, but the book uses the mapping in **Table 2.1**.

**Sponsored products: the advertised product**  
The retrieved candidate for sponsored products is an **advertised product**, an (SKU, ad group) pair, because one SKU can appear in several ad groups. A brand might advertise the same coffee maker in an always-on ad group with auto-targeting and a low bid, and in a holiday ad group with manual keywords and a higher bid. Each pair must reach the auction with its own bid and charge its own budget, while the auction and page arbitration keep at most one copy of the SKU on the page (Chapters 6 and 8). **Figure 2.1** shows both branches of the hierarchy.

**Figure 2.1: Campaign Entity Hierarchy for Sponsored Products and Display**

```mermaid
flowchart TB
  ADV["Advertiser account"] --> C1["Campaign (sponsored products): objective, budget, flight, bid strategy"]
  ADV --> C2["Campaign (display): objective, budget, flight"]
  C1 --> AG1["Ad group: keywords or auto-targeting, bids"]
  C1 --> AG2["Ad group: product targeting, bids"]
  AG1 --> A1["Ad: SKU 1"]
  AG1 --> A2["Ad: SKU 2"]
  AG2 --> A3["Ad: SKU 1"]
  C2 --> LI["Ads line item: targeting, placements, flight, impression goal or budget, price, priority tier, frequency cap"]
  C2 --> IO["Insertion order: contract for a guaranteed or managed buy"]
  IO --> LI
  LI --> CR1["Ad: creative"]
  LI --> CR2["Ad: creative"]
  A1 -.->|"advertised product (SKU 1, ad group 1)"| AU["Auction"]
  A3 -.->|"advertised product (SKU 1, ad group 2)"| AU
  CR1 -.->|"ads line item with its selected creative"| AU
```

> **Design trade-off: One entity model for both formats vs. a separate display schema.** One model, with format-specific fields on the ad group, lets one control plane, index service, budget service and reporting pipeline serve both formats, at the cost of nullable fields and per-format validation. Separate schemas keep each model simple but duplicate infrastructure and make cross-format pacing and page arbitration harder. Choose one model when display is built on the sponsored-products stack in Stage 3; keep separate schemas only while display runs on a third-party ad server (Chapter 4 makes the matching serving decision).

> **Design trade-off: Advertised products vs. SKUs as the retrieval unit.** Retrieving advertised products carries each ad group's targeting, bid and budget through ranking, so the auction prices the right bid and charges the right budget; it doubles the index in the reference RMN (10 million pairs for 5 million SKUs) and requires deduplication before the page. Retrieving SKUs keeps the index small and scores relevance once per item, but forces the platform to choose among a SKU's ad groups after retrieval. A middle path indexes shared content once per SKU and expands each retrieved SKU into its eligible advertised products before eligibility filtering and bidding; Chapter 4 and Appendix 1 develop the index layout.

**Other entities**  
A **creative** is the asset a shopper sees: a product tile assembled from catalog data, a banner, a video. A **placement** is a location on a page where ads render. An **impression** is counted at one of three stages: *served* (the ad server returned the ad), *rendered* (the client displayed it) or *viewable*. The Media Rating Council's standard counts a display impression as viewable when at least 50% of its pixels are in view for at least one second, and a video impression when 50% are in view for at least two seconds [3].

### 1.2. Glossary of Core Terms

**Table 2.2** defines terms that recur in every chapter but have no home section of their own. Formats and pricing are defined in Section 4.2, targeting in Section 4.3, budgets and bid strategies in Section 5 and metrics in Section 6; `glossary.md` gives full definitions of all of them.

**Table 2.2: Core Terms Without a Home Section**

| Term | Meaning | Developed in |
|---|---|---|
| **First-party data** | Data the retailer collects from its own shoppers: searches, views, carts, purchases, loyalty records | Ch11, Ch15 |
| **Pseudonymized ID** | A tokenized user identifier used in logs and features in place of a direct identifier; re-identifiable only with separately held information | Ch15 |
| **Hashed identifier** | An email address or phone number, normalized (lowercased and trimmed; phone numbers in E.164 format) and passed through a one-way hash such as SHA-256 [4], so that two parties can find shared customers without exchanging raw contact details. Hashes can still identify a person, so they are not anonymous [5] | Ch15, A4 |
| **Match rate** | The share of one party's records matched to the other party's identities. It bounds the reach of an activated audience and the coverage of a measurement study | Ch15, Ch17 |
| **Clean room, identity resolution** | Defined in Chapter 1, Section 3.3 | Ch15 |
| **eCPM** | Expected revenue per thousand impressions, the common currency of the auction (**Equation 1.1**) | Ch6 |
| **Reserve price** | The minimum price a candidate must clear to take part in the auction | Ch6, Ch9 |
| **Frequency cap** | The maximum number of exposures per shopper per period for an ads line item | Ch7, Ch16 |
| **Ad load** | How many ad slots a page carries, and where (Section 5.2) | Ch8 |

---

## 2. The System Map and a Reference RMN

### 2.1. The End-to-End System Map

Chapter 1 drew the RMN as a business (**Figure 1.2**) and as a position in the advertising ecosystem (**Figure 1.3**). **Figure 2.2** is the operating view used throughout the rest of the book: how advertiser changes become serving state, how one ad request moves through the onsite stack, and how events come back as billing, reporting and learning signals. Later chapters unpack the services, latency budgets and failure behavior in detail.

**Figure 2.2: The RMN Operating Map: Control Plane, Serving Path and Measurement Loop**

```mermaid
%%{init: {'themeVariables': {'fontSize': '24px'}, 'flowchart': {'nodeSpacing': 24, 'rankSpacing': 36}}}%%
flowchart LR
  ADV["Advertiser"]
  SHO["Shopper on site or app"]
  PRT["Partner DSPs and publishers"]

  subgraph CP["Advertiser control plane"]
    CON["Console and APIs"] --> MGT["Setup, policy, review"]
    CAT["Catalog and retail state"] --> CFG["Serving config and eligibility"]
    MGT --> CFG
  end
  subgraph SP["Onsite serving path"]
    REQ["Ad request: query/page, slot, shopper"]
    DRQ["Display request: placement, no query"]
    CND["Candidate generation and eligibility"]
    DLI["Ads line item rules and eligibility"]
    SEL["Scoring, auction, selection"]
    DEL["Page decision, creative delivery"]
    REQ --> CND
    DRQ --> DLI
    CND --> SEL
    DLI --> SEL
    SEL --> DEL
    PAC["Budget and pacing state"]
    ABD["Autobidding"]
    PAC -.->|"throttle participation"| CND
    PAC -.->|"budget multiplier"| ABD
    ABD -.->|"set bids"| SEL
  end
  subgraph DP["Measurement and feedback"]
    EVT["Impression, click, purchase events"] --> QLT["Traffic quality, billing, reporting"]
    QLT --> MEA["Attribution and incrementality"]
    QLT --> TRN["Training and optimization signals"]
  end
  subgraph OF["Offsite activation and data products"]
    AUD["Audiences and clean rooms"]
  end

  ADV --> CON
  SHO --> REQ
  SHO --> DRQ
  DEL --> SHO
  MEA -.->|"reports and insights"| ADV
  CFG --> CND
  CFG --> DLI
  DEL --> EVT
  QLT -.-> PAC
  TRN -.-> SEL
  TRN -.-> AUD
  AUD --> PRT
  PRT -.->|"offsite exposures"| MEA
```

*Solid lines: configuration, requests and delivery. Dashed lines: feedback from events and state.*

**The control plane**  
Advertisers create and change campaigns through a console or APIs. The control plane validates each change, sends new creatives through ad review (Chapter 10), and joins advertised SKUs with catalog state, such as price, stock and eligibility, to build the serving configuration. Chapter 3 takes up how those changes propagate safely.

**The serving path**  
A sponsored-products request follows the ranking cascade of Chapter 1, Section 6, from candidate generation and eligibility, through scoring and pricing, to the page decision and creative delivery. Budget pacing acts at two points: throttling decides during candidate generation whether a campaign takes part, and bid scaling adjusts an autobid campaign's bid in the auction (Section 5.1). Chapters 3 through 8 decompose that path stage by stage.

**Where display branches off**  
A display request comes from a placement, such as a homepage hero slot, and carries no query. Its candidates are ads line items whose targeting rules match the page and the shopper, subject to delivery and frequency rules; guaranteed commitments may be allocated before the auction, and the rest uses the same serving stack. Display and sponsored products then compete for the same page in page arbitration (Chapter 8).

**Where offsite branches off**  
Offsite media does not use the onsite serving path. The RMN activates audiences on partner DSPs and publishers through clean rooms, and offsite exposures return to the measurement systems to be matched with sales (Chapters 15 and 17).

**The measurement loop**  
Every served ad, click and purchase becomes an event, and those events drive billing and budget state, attribution (Chapter 13), incrementality measurement (Chapter 14) and the training or optimization signals used later in the stack (Chapters 5 and 11). Reports flow back to advertisers, and the loop closes in the control plane.

### 2.2. A Reference RMN

Numbers in an architecture book only help when they agree: a retrieval budget must fit the end-to-end budget, and a cascade size must fit the traffic it sustains. The book therefore uses one **reference RMN**: a **fictional composite** retailer at the scale of the largest US retail media networks, built from public reporting and round assumptions. It does not describe any company. For scale, Walmart reported that its global advertising business grew 46% in fiscal 2026 to nearly \$6.4 billion, including VIZIO [1]; the reference RMN's onsite advertising revenue of about \$5 billion is of that order. **Table 2.3** gives the profile, for the US onsite business at peak season.

**Table 2.3: The Reference RMN Profile**

| Quantity | Value | Basis |
|---|---|---|
| Annual onsite ad revenue | ~\$5B | Public scale of the largest US RMNs, rounded |
| Monthly active online shoppers | ~120M | Assumption |
| Known customer profiles | ~200M, online and in-store loyalty; used for closed-loop measurement | Assumption |
| Monthly sessions, web and app | ~800M | Assumption |
| Catalog | ~400M SKUs, first-party and marketplace | Assumption |
| Advertised products | ~10M (SKU, ad group) pairs over ~5M distinct SKUs | Assumption |
| Ad groups | ~1M | Derived: ~10 advertised products per ad group |
| Advertiser keyword targets | ~30M (keyword, ad group) pairs | Assumption |
| Active advertisers | ~50K suppliers, marketplace sellers and agencies | Assumption |
| Active display ads line items | ~10K, about 5% guaranteed | Assumption |
| Ad requests | Design point: ~15K/s average (~1.3B/day), ~50K/s peak | Derived below, with headroom |
| Sponsored impressions | ~3B/day | Derived below |
| Post-click conversion rate; attributed order value | ~10%; ~\$35 | Assumption (Ch5) |
| End-to-end ad latency (p99) | Search-integrated ≤ 60 ms; item and home placements ≤ 100 ms; asynchronous display ≤ 150 ms | Design budget (Ch3) |
| Stage latency budgets (p99) | Retrieval, including throttling and budget eligibility, ≤ 15 ms; ranking stage (relevance gate and CTR/CVR model) ≤ 20 ms for ~300 candidates; auction, bid scaling and pricing ≤ 5 ms; page arbitration ≤ 5 ms | Design budget (Ch3) |
| Ranking cascade | ~3,000 merged candidates → ~300 after the pre-ranker → ~200 after the relevance gate → top ~30 to the auction → ~6 sponsored slots on a search page, ~8 in an item-page carousel | Design budget (Ch1 **Figure 1.4**; Ch4, Ch5) |

The rows relate to each other in three ways, and later chapters rely on the same arithmetic.

**Traffic**  
About 800 million sessions a month at about 4.7 page views per session is about 3.8 billion page views a month, or about 1,400 page views per second on average. Each page view issues about 4 ad requests, one for each of several placements plus prefetching for the next screen. That gives about 6,000 ad requests per second on a typical day, or about 500 million a day, and a holiday peak runs 3 to 4 times higher. The profile's design point of 15,000 requests per second on average and 50,000 at peak is about 2 to 3 times the derived load, deliberate headroom so a traffic spike or regional failover does not break the latency budget. Chapter 3 sizes the serving fleet against the design point.

**Revenue**  
About 500 million ad requests a day, each serving about 6 sponsored ads on average, is about 3 billion sponsored impressions a day. The average is 6 because a search page fills about 6 slots and every product in an item-page carousel counts. \$5 billion a year is about \$13.7 million a day, so the effective CPM is about \$4.6. That is consistent with an average cost per click of about \$1 and a click-through rate of about 0.5% per served impression, which by **Equation 1.1** gives an eCPM of \$5. The click-through rate per served impression is low because many carousel positions are served but never scrolled into view. At that click-through rate the network sees about 15 million clicks a day. With a post-click conversion rate of about 10% and an average attributed order value of about \$35, each click returns about \$3.50 in attributed sales for about \$1 of cost, an attributed ROAS of about 3.5. Section 6.3 explains why that figure overstates what the ads caused.

**Catalog and demand**  
About 5 million of the 400 million SKUs are advertised, each in two ad groups on average: about 10 million advertised products (Section 1.1), and at about 10 per ad group, about 1 million ad groups. The retrieval index (Chapter 4) is sized by advertised products and keyword targets, not by the whole catalog.

---

## 3. The Advertiser Lifecycle

An advertiser's work on an RMN is not a one-time launch. It is a control loop with three phases: planning, execution and measurement (**Figure 2.3**). Each day the platform solves the same allocation problem: which products to promote, against which demand, under which budget and return constraint, and with what evidence that the spend paid off.

In planning, the advertiser chooses the commercial objective first and the platform objects second. A supplier launching a new SKU may accept a lower short-term ROAS to buy visibility. A mature endemic brand may care more about efficient share capture on branded and category queries. A non-endemic advertiser may care about reach and frequency on premium placements. Those choices determine the format, targeting, creative requirements, budget shape and bid strategy. They also determine the measurement standard later.

Execution is where configuration becomes online state: campaigns go live, pacing spends the budget, auctions price each opportunity, and reports expose enough signal to react. Measurement closes the loop by deciding what the spend accomplished. Sections 4 to 6 unpack that machinery.

**Figure 2.3: The Advertiser Lifecycle as a Continuous Loop**

```mermaid
graph LR
    A["Business goals: launch a product, reach a ROAS target"]
    B["Planning: formats, targeting, creatives, budgets, bids"]
    C["Launch: campaign goes live"]
    D["Execution: auctions, pacing, in-flight changes"]
    E["Measurement: attributed results, incrementality, insights"]
    F["Refinement: apply learnings to the next cycle"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F -.Feedback loop.-> A

    style A fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style C fill:#fff3cd,stroke:#856404,stroke-width:2px
    style E fill:#d4edda,stroke:#28a745,stroke-width:2px
    style F fill:#f8d7da,stroke:#721c24,stroke-width:2px,stroke-dasharray: 5 5
```

**Table 2.4** maps each phase to the activities, metrics and platform components involved.

**Table 2.4: Advertiser Lifecycle Phases and Platform Components**

| Phase | Activities | Key metrics | Platform components | Section |
| --- | --- | --- | --- | --- |
| **Planning** | Set goals; choose formats, products and audiences; build campaigns; create creatives; set budgets and bids | Forecast reach, estimated ROAS | Campaign management, creative management, targeting and audiences, forecasting | 4 |
| **Execution** | Launch; monitor delivery; change bids, budgets and targeting; test creatives | Impressions, clicks, CTR, spend against pacing, ROAS to date | Ad serving, auction, budget pacing, autobidding, real-time reporting | 5 |
| **Measurement** | Analyze ROAS and CPA; read attribution breakdowns; run lift studies; plan the next cycle | Final ROAS and CPA, incremental ROAS, new-to-brand share | Reporting, attribution, experimentation platform | 6 |

---

## 4. Ads Planning: Campaign Setup

### 4.1. Campaign Management

Campaign setup is the translation layer between marketing intent and serving state. The campaign holds the business objective, budget envelope, flight and scheduling rules; ad groups partition that intent into units that can be controlled independently, by product family, match strategy or margin tier for sponsored products, or by placement for display.

The setup model must be simple enough for an advertiser to reason about and specific enough for the serving stack to enforce online. A budget attached only at campaign level is easy to explain but too blunt for bidding control, while an ad-group structure that mirrors every merchandising nuance is precise but impossible to maintain. Good campaign management therefore exposes a limited set of stable controls and pushes the harder optimization into the platform.

Forecasting sits between setup and launch. The advertiser needs a plausible estimate of reachable demand, bid levels and likely spend, and the platform builds it from historical demand, current competition, targeting reach and the bid response curves of Chapter 9. Once launched, the control plane validates the configuration and publishes it into serving state within the propagation targets of Chapter 3, Table 3.6. For self-serve sponsored products, that publication path has to reject broken configuration without feeling slow or batch-like.

### 4.2. Ad Formats: Sponsored Products vs. Display

An **ad format** is the kind of unit the shopper sees, such as a catalog-native sponsored product or a display creative. A **placement** is where that unit appears on a page. These are separate dimensions: one surface can host different formats, and one format can appear on several surfaces. Most RMN inventory falls into two format families, with one hybrid.

**Sponsored products**  
A **sponsored product** is a promoted catalog SKU rendered natively in shopping surfaces: at the top of, alongside or within search results, and on product pages [6]. In the reference RMN used throughout this book, the canonical sponsored-products unit is assembled from catalog data such as image, title, price, brand and rating, so the advertiser decides which SKUs to promote, which demand to target and how aggressively to bid. Sponsored products are usually priced per click and sold by auction [6]. The retrieved unit is the advertised product of Section 1.1.

**Display**  
A **display ad** is an image, video or HTML unit that occupies a dedicated slot: a homepage hero, a category banner, a product-page banner, or a search-results brand placement. Commercially, display usually trades on impressions rather than clicks and can be sold either through the auction or through guaranteed deals.

**Brand units**  
A **brand unit** sits between the two. It lives on a search page, is targeted like sponsored products, and is authored like display: logo, headline and multiple products in one composite placement.

> **Sidebar (as of October 2026): Brand units in practice.** Amazon's Sponsored Brands combine a brand logo, a headline, a video or lifestyle image and multiple products; they appear on top of and within shopping results, on product pages and on the homepage; they are targeted by keywords or products; and they are priced per click or per thousand viewable impressions [7].

The distinction is easier to see on real placements:

1. A sponsored product: a coffee maker listed in the results for "coffee maker", charged per click.
2. Display, guaranteed: a homepage hero banner for a coffee brand's seasonal line, sold for one week at a fixed price per thousand impressions.
3. Display, non-endemic: a credit-card offer on the checkout page, bought by a bank that does not sell through the retailer (Chapter 1, Section 4.1).
4. A brand unit: the coffee brand's logo, headline and three of its products at the top of the results for "ground coffee".

> **Sidebar (as of October 2026): Walmart examples of onsite placements.** Live retail surfaces make the format distinction concrete, but they are not stable taxonomy documents, so they belong in a dated example. Walmart search results for `coffee maker` show both a top-of-results sponsored brand unit and in-grid product cards explicitly labeled `Sponsored`, which are clean sponsored-products examples [8]. Walmart item pages can also contain sponsored recommendation or offer modules within the product-detail experience, as visible lower on the Mr. Coffee 5-Cup product page [9].

**Figure 2.4: Walmart search results show both a top sponsored brand unit and in-grid sponsored product cards for `coffee maker` (October 2026)**

<img src="resources/ch02/ch02_walmart_search_composite.png" alt="Two Walmart search-result screenshots for coffee maker side by side: on the left, a sponsored Keurig brand unit at the top of the results; on the right, in-grid product cards labeled Sponsored." width="100%">

<br><br>

**Figure 2.5: Walmart item pages can pair the full product-detail context with lower sponsored recommendation and offer modules (October 2026)**

<img src="resources/ch02/ch02_walmart_item_composite.png" alt="Two Walmart product-page screenshots side by side: on the left, the Mr. Coffee 5-Cup product page hero; on the right, a lower section of the same page with a sponsored recommendation carousel and a sponsored offer module." width="100%">

<br><br>

**Table 2.5** compares the two primary families.

**Table 2.5: Sponsored Products vs. Display**

| | Sponsored products | Display |
|---|---|---|
| **What is advertised** | A specific SKU | A brand, a product line, an offer, or (non-endemic) an outside product or service |
| **Unit retrieved and served** | Advertised product: an (SKU, ad group) pair | Ads line item, then one of its creatives |
| **Creative** | Assembled from catalog data | Produced by the advertiser or generated with AI tools (Ch10) |
| **Typical placement** | Inline in search results; product-page and cart carousels | Homepage hero, category and product-page banners, brand units on search results, video |
| **Targeting** | Manual keywords, auto-targeting, products and categories | Audiences, context, categories, retargeting |
| **Typical pricing** | CPC | CPM or vCPM (CPC possible) |
| **Buying model** | Self-serve auction | Auction, or guaranteed and sponsorship deals |
| **Measurement** | Click-based attribution, ROAS | View-through attribution, reach and frequency, lift studies, media mix models |
| **Typical buyer** | Endemic suppliers and marketplace sellers | Endemic brands (for awareness) and non-endemic advertisers |

**Format, placement, pricing model and buying model**  
The book keeps these axes separate because they answer different design questions. **Format** says what kind of ad unit is rendered. **Placement** says where it appears; a product page can carry both a sponsored-products carousel and a display banner. The **pricing model** says what event creates the charge: a click (CPC), an impression (CPM), or a viewable impression (vCPM), where viewability follows Section 1.1. The **buying model** says how the inventory is sold: in the auction, or through **guaranteed delivery**, where the RMN commits to an impression goal within a flight at an agreed price. A **sponsorship** is a guaranteed, often exclusive version of that buy. The pairings in **Table 2.5** are common, not mandatory: a brand unit can trade on clicks or viewable impressions, and display can be auctioned or reserved. Pricing changes what the auction must predict (Chapter 6); the buying model determines whether there is an auction at all.

> **Design trade-off: Auction vs. guaranteed buying for display.** Guaranteed deals give advertisers certainty about delivery, placement and price, and give the RMN predictable revenue; they require inventory forecasting, delivery-goal pacing, make-goods and reserving inventory the auction might have sold for more. The auction needs none of that machinery and prices each impression at its current value, but cannot promise delivery. Sell premium placements (homepage heroes, tentpole events) as guaranteed or sponsorship deals once forecasting exists, and run the remaining display inventory through the auction alongside sponsored products (Chapters 7 and 16).

**The coverage principle**  
Part II uses sponsored products as the main thread and then states what changes for display in each chapter's "Display Placements: What Changes" subsection. Chapter 16 takes the display business in its own right, including video as a display subtype; Chapter 17 handles offsite media; Chapter 10 handles creative systems for both formats.

### 4.3. Targeting and Audience Strategy

Targeting decides whether an ad is *eligible* for a request. It has two separate dimensions: the placement where the ad may appear, and the rule by which the platform decides that the request qualifies. For sponsored products, each major targeting type maps to a retrieval path in Chapter 4.

#### 4.3.1. Placement Types

The placement sets the shopper's intent, the latency budget and often the economics of the ad unit. The examples below describe surfaces and typical formats, not one-to-one rules.

- **Search results:** sponsored products in the grid, brand units above it, display at its edges. Intent is highest because the shopper typed a query. The budget is the tightest, 60 ms end to end in the reference RMN, because the ads render inside the results page.
- **Browse and category pages:** the shopper is exploring a category without a specific query; intent is moderate.
- **Product pages:** carousels of substitutes or complements for the viewed item, found by product-to-product retrieval (Chapter 4). Intent is high but narrow; the budget is 100 ms.
- **Homepage, cart and checkout:** awareness units, add-on suggestions and checkout offers, where display is common; asynchronous display slots may take up to 150 ms.
- **Offsite:** display and video on other publishers' inventory, targeted with the retailer's audiences (Chapter 17).

#### 4.3.2. Targeting Mechanisms

**Manual keyword targeting**  
The advertiser chooses keywords and a **match type** for each [10]. *Exact* match requires the query to equal the keyword; *phrase* match requires the query to contain the keyword's words in order; *broad* match allows the words in any order and admits plurals, variations, synonyms and related terms. **Close variants** are query forms the platform treats as matches even under the stricter match types: misspellings, plural forms, stems, abbreviations and reorderings [11]. Broad match and close variants are therefore implemented by rewriting the query, not by literal token matching (Chapter 1, Section 1.3). **Negative keywords** exclude an ad from queries that contain them [10].

**Auto-targeting**  
With **auto-targeting**, the advertiser supplies no keyword list. The platform infers matching queries and placements from the product itself and from similar or complementary items, which makes retrieval quality the platform's responsibility.

> **Sidebar (as of October 2026): Match types and auto-targeting on two large platforms.** On Google Ads, every keyword match type is eligible to match close variants, and advertisers cannot opt out [11]. Amazon divides sponsored-products automatic targeting into four groups that can be bid separately: close match and loose match (search terms closely or loosely related to the product), and substitutes and complements (detail pages of similar or complementary products) [10].

**Product and category targeting**  
**Product targeting** places ads on or against specific products, categories, brands or product features [10]. Typical uses are conquesting on a competitor's product page, cross-selling on complementary products, or narrowing a category target.

**Audience, contextual and retargeting**  
These mechanisms are native to display and also serve as overlays on sponsored products. **Audience targeting** makes an ad eligible for shoppers in a segment defined over first-party data: behavioral ("searched running shoes in the past 7 days and did not buy"), transactional ("bought a competing brand three times in 90 days") or predictive ("likely to buy in this category this week", Chapter 11). In the reference RMN, advertisers see segment sizes and results, never shopper-level data. **Contextual targeting** matches the page rather than the shopper, so it needs no identity, and it can be extended with embedding similarity (Chapter 4). **Retargeting** brings back shoppers who recently viewed or carted a product without buying.

**Boolean targeting**  
An ad group's targeting is a **Boolean** expression over these mechanisms, for example keyword AND audience AND NOT category. The platform indexes those expressions in disjunctive normal form so one request can be matched against millions of eligibility rules at once. That index is the eligibility layer for sponsored products and the core of display retrieval (Chapter 4). **Table 2.6** summarizes the mechanisms.

**Table 2.6: Targeting Mechanisms and Where They Are Served**

| Mechanism | Formats | Who chooses the matches | Served by (Ch4) |
|---|---|---|---|
| Manual keyword (exact, phrase, broad; negatives) | Sponsored products, brand units | Advertiser, widened by close variants and query rewriting | Lexical retrieval over keywords and query rewrites |
| Auto-targeting | Sponsored products | Platform | Lexical and embedding-based retrieval over product content |
| Product and category | Sponsored products, brand units, display | Advertiser | Product-to-product retrieval; category eligibility |
| Audience and retargeting | Display; overlay on sponsored products | Advertiser chooses segments; platform builds predictive ones (Ch11) | Eligibility filter over segment membership |
| Contextual | Display | Advertiser | Eligibility filter; embedding-based contextual matching |

> **Design trade-off: Manual keyword targeting vs. auto-targeting.** Manual keywords give advertisers control and predictability: they know which queries they buy and can exclude the rest. They also need constant maintenance and miss queries nobody thought of. Auto-targeting covers the long tail with no setup and makes small sellers viable advertisers, but shifts relevance responsibility to the platform, and advertisers trust it only with transparent search-term reports and negative keywords as a backstop. Offer both from Stage 1, default new advertisers to auto-targeting, and let manual keywords refine or override it. Auto-targeting is only as good as the embedding-based and product-to-product retrieval behind it (Chapter 4).

#### 4.3.3. Common Combinations

Real campaigns usually layer mechanisms, pairing a placement with several signals at once (**Table 2.7**).

**Table 2.7: Common Placement and Targeting Combinations**

| Placement | Typical targeting | Example |
| --- | --- | --- |
| Search results | Keywords or auto-targeting, with an audience overlay | Bid on "protein bar" with a higher bid for shoppers with three or more past snack purchases |
| Product page | Product targeting, audience | Show a carousel ad on competitors' product pages to shoppers who bought the brand before |
| Category page | Category (contextual) | Banner on "Sports & Outdoors" for a running-shoe launch |
| Homepage | Audience | Retarget shoppers who left a product in their cart |
| Offsite | Audience through a clean room | Reach past buyers of a brand on connected TV |

---

## 5. Running Campaigns

Once campaigns are live, three subsystems run them in real time: budgets, auctions and the controls through which advertisers and the platform change campaigns in flight.

### 5.1. Budget Management

Advertisers set **daily** budgets, **lifetime** budgets for a flight, optional ad-group budgets and **portfolio** budgets that share spend across multiple campaigns. In the UI, those controls look simple. In serving, they create a distributed state problem: any machine may receive the next request, yet the platform has to decide in milliseconds whether spending keeps the campaign on track or pushes it toward overspend.

The core engineering problem is **pacing**. A campaign should neither exhaust its budget by mid-morning nor end the day with budget unspent. The control lever depends on the buying model. Manual-bid campaigns and guaranteed ads line items are usually paced by **throttling** participation: the platform decides whether the campaign enters the candidate set for a given request. Autobid campaigns are often paced by **bid scaling**: the campaign remains eligible, but its bid is multiplied by a budget-aware factor. Guaranteed ads line items invert the failure mode: their main risk is under-delivery against an impression goal rather than overspend against a spend cap.

Budgets therefore act on serving as well as on finance: they change retrieval eligibility, auction participation, reporting semantics and advertiser expectations. Chapter 7 owns the online state and feedback control behind those decisions.

### 5.2. Bidding and Auctions

Each ad request triggers an auction. Candidates are compared in **eCPM**, the expected revenue per thousand impressions. CPC candidates rank on the eCPM of **Equation 1.1**, the bid times the predicted click-through rate times 1,000; CPM candidates rank on the bid itself:

$$
\text{eCPM} = \text{CPM}_{bid}
\tag{2.1}
$$

Campaigns bidding for conversion value, such as those with a target ROAS, add one step: the platform predicts the value of a click or impression from pCVR and expected order value, and converts it into an auction bid subject to the campaign's target. The auction still ranks a bid-derived eCPM, not raw expected order value. Chapter 9 develops the bidder logic in detail.

RMNs mostly price their auctions with the **generalized second-price (GSP)** rule: each winner pays the minimum it needed to keep its position. Unlike the Vickrey-Clarke-Groves mechanism, GSP does not make truthful bidding a dominant strategy [12], so a bid is not a clean statement of advertiser value (Chapter 1, Section 1.3). Chapter 6 therefore adjusts the ranking for quality, relevance and reserve prices.

The auction also sits downstream of retrieval and upstream of pacing and billing. By the time a candidate reaches this stage, retrieval and the relevance gate have already decided that it is relevant enough, and the budget system has already decided that the campaign may spend. After the auction, the winner creates cost events, budget depletion, attribution credit and training examples. Chapters 5, 6, 7, 9 and 13 all touch the auction even though the pricing rule itself belongs to Chapter 6.

**Auction and guaranteed delivery on one page**  
Guaranteed ads line items do not bid. Their impressions are allocated before the auction to meet the contracted goals (Chapter 16), and the remaining inventory is auctioned. Sponsored products, display and organic results then compete for the same page. How many ad slots a page carries, and where, is its **ad load**, which Chapter 8 owns along with page arbitration.

**Bid strategies**  
A **bid strategy** determines who sets the bid for each auction and what the platform pursues on the advertiser's behalf (**Table 2.8**).

**Table 2.8: Campaign Goals and Bid Strategies**

| Campaign goal | Typical bid strategy | Who sets the bid | Platform's role |
| --- | --- | --- | --- |
| Awareness (reach) | Manual CPM | Advertiser | Auction only |
| Consideration (traffic) | Manual CPC | Advertiser | Auction only |
| Consideration or sales | Dynamic bidding (CPC) | Advertiser sets a base bid; the platform adjusts it | Raises or lowers the bid per auction with the predicted conversion likelihood |
| Sales at a cost target | Autobidding: target CPA | Platform | Sets each bid to reach the average cost per conversion |
| Revenue at a return target | Autobidding: target ROAS | Platform | Sets each bid to reach the average return on ad spend |
| Volume within budget | Autobidding: maximize conversions | Platform | Spends the budget for the most conversions |

**Dynamic bidding** modifies a manual bid: the advertiser sets a base CPC, and the platform raises or lowers it per auction within a bounded range using the predicted likelihood of a sale. **Autobidding** removes the per-auction bid entirely: the advertiser states a goal and a constraint, and the platform computes a bid for every auction from predicted conversion probability and value. Autobidding depends on calibrated pCVR and value predictions (Chapter 5); Chapter 9 develops bid strategies for both formats.

> **Sidebar (as of October 2026): Dynamic bidding and autobidding on two large platforms.** Amazon's "dynamic bids, up and down" strategy for sponsored products raises or lowers bids by up to 100%, so a \$1.00 bid can become at most \$2.00 [13]. Google's target CPA strategy sets a bid each time an ad is eligible, from historical campaign data and auction-time signals [14], and its target ROAS strategy predicts the conversion value of each search and bids to reach the advertiser's average return target [15].

### 5.3. Advertiser Controls and In-Flight Changes

**Table 2.9** lists the controls an advertiser holds and the chapter that owns the system behind each. Later chapters point here rather than list them again.

**Table 2.9: Advertiser Controls and Their Owning Chapters**

| Control | Examples | Set on | Owning chapter |
|---|---|---|---|
| **Budgets** | Daily, lifetime, ad group, portfolio | Campaign, ad group | Ch7 |
| **Schedules** | Flight dates, dayparting | Campaign, ads line item | Ch7 |
| **Bid strategies** | Manual CPC or CPM, dynamic bidding, target CPA, target ROAS, maximize conversions | Campaign, ad group | Ch9 |
| **Goals** | CPA or ROAS targets, conversion values | Campaign | Ch9 |
| **Targeting** | Keywords, match types, negatives, auto-targeting groups, products and categories | Ad group | Ch4 |
| **Audiences** | Segments, predictive audiences, retargeting, customer lists | Ad group, ads line item | Ch11, Ch15 |
| **Delivery terms** | Impression goal, priority tier, frequency cap, placements | Ads line item | Ch7, Ch16 |
| **Creatives** | Display assets, brand-unit headlines, catalog content for sponsored products | Ad, ads line item | Ch10 |
| **Experiments** | Split tests of bids, budgets, targeting or creatives | Campaign | Ch14 |

Advertisers steer the campaign while it is live: raising bids where demand is profitable, excluding wasteful queries, rotating creatives, or splitting a broad ad group. The platform increasingly automates the same decisions through autobidding, auto-targeting and creative selection (Chapter 10), and at the limit an automated campaign asks for a budget, a goal and a product set, then infers most of the rest. When the platform makes the tactical moves, the advertiser needs explanations and experiments in addition to dashboards of clicks and spend.

---

## 6. Measurement and Attribution

Measurement closes the loop of Section 3 by answering three different questions. First, what happened operationally: impressions, clicks, spend, attributed orders and pacing status. Second, what happened commercially: reach, frequency, ROAS, CPA and new-to-brand acquisition. Third, what happened causally: which outcomes the ads actually changed.

The measurement surface depends on the format. Sponsored products are usually judged on click-based outcomes because the ad unit sits close to purchase intent. Display, especially for non-endemic advertisers, puts more weight on views, reach, frequency and lift because the ad often serves earlier in the funnel or for a business whose sale happens elsewhere. Both cases depend on **closed-loop measurement**, which links exposures to real purchases, online and in store, through the retailer's known customer base. That link joins impression and click logs to order events, loyalty identities and cost records. Chapter 13 builds that system inside the retailer, Chapter 14 adds the causal methods, and Chapter 15 handles cases where the relevant exposure or conversion crosses the retailer boundary.

### 6.1. Core Metrics

**Table 2.10** defines the metrics later chapters use, with the format each mainly serves.

**Table 2.10: Core RMN Performance Metrics**

| Metric | Formula | What it measures | Mainly for |
| --- | --- | --- | --- |
| **ROAS** | `attributed_revenue / ad_spend` | Attributed revenue per dollar spent; the headline advertiser metric | Sponsored products |
| **CPA** | `ad_spend / conversions` | Cost per attributed conversion | Both |
| **CTR** | `clicks / impressions` | How often an ad is clicked; reflects relevance, creative and position | Both |
| **CVR** | `conversions / clicks` | How often a click leads to a purchase | Sponsored products |
| **New-to-brand** | `first_time_buyers / total_buyers` | Customer acquisition rather than repeat purchase | Both |
| **Reach, frequency** | `unique_shoppers_exposed`; `impressions / unique_shoppers_exposed` | How many shoppers saw the ads, and how often | Display |
| **Incremental share** | `(conv_rate_treatment − conv_rate_control) / conv_rate_treatment` | The share of the exposed group's conversions that the ads caused | Both (Section 6.3) |
| **iROAS** | `incremental_revenue / ad_spend` | Revenue caused per dollar spent | Both (Section 6.3) |

### 6.2. Attribution Models

**Attribution** assigns the credit for observed conversions to the ads a shopper saw or clicked before them.

- **Last-touch** gives all credit to the last ad interaction before the purchase. It is the common default on RMNs because it is simple and deterministic when exposure and purchase share an account.
- **Multi-touch attribution (MTA)** splits credit across the ads a shopper interacted with, by a fixed rule (linear, time decay, position based) or with learned weights, but still does not measure what the ads caused (Chapter 14 reconciles the two).
- **Attribution windows** limit how long after an interaction a purchase can be credited, shorter for views than for clicks; in-store purchases need a window of days because they are matched through loyalty accounts.
- **Click-through vs. view-through:** sponsored products are attributed mostly on clicks; display on views as well, with shorter windows and often lower weight, because a view is weaker evidence of influence than a click.

Attribution is fundamentally a log-joining problem with policy layered on top: which exposure records are eligible, which purchase records match, which time window applies, and how the reported number should be interpreted. On an RMN, those joins are stronger than on an open-web system because the retailer often owns both the exposure and the conversion. Chapter 13 builds the event joins, attribution rules, late-arriving conversion handling and reporting pipelines. Attribution assigns credit to observed outcomes; it does not prove that the ad caused them.

### 6.3. Incrementality and Experimentation

Attribution and incrementality answer different questions. Attribution, including MTA, divides the credit for conversions that happened. **Incrementality** asks what changed because the ad ran: the difference between the observed world and the counterfactual world in which that exposure did not happen. For an RMN, that gap is the central credibility problem because the platform reaches shoppers who are already intent on buying. A shopper who clicked a sponsored listing and bought the item may look like an advertising success and still represent little true lift if the same shopper would have bought organically a few seconds later.

That is where **cannibalization** enters. Sponsored products can shift credit from organic commerce to paid media without creating much extra demand. The more the ad system monetizes high-intent traffic, the more advertisers ask whether the platform is capturing sales they would have received anyway. Field evidence from search advertising shows that the gap can be large: at eBay, Blake, Nosko and Tadelis found that the returns from paid search were a fraction of conventional non-experimental estimates and that brand-keyword ads had no measurable short-term benefit [16]. RMNs face the same scrutiny, often more sharply, because the ad is shown inside the store.

The operational answer is experimentation. Hold out a portion of shoppers or queries from exposure, then compare outcomes. Closed-loop measurement makes that easier for RMNs than for most publishers because the retailer sees both exposure and purchase. It does not make it free: holdouts reduce monetization, small campaigns produce noisy estimates, and cross-channel interference complicates interpretation. Even so, once the ads business becomes material, attributed ROAS alone is not enough. Advertisers want **incremental ROAS (iROAS)**, and the platform needs that causal signal to inform ranking, bidding, audience selection and product strategy.

Chapter 14 owns the methods in detail: the experimentation platform, holdout and geo designs, predicted lift where no experiment ran, and reconciliation between attribution, experiments and media mix models. Chapters 6, 9 and 11 use incremental value in ranking, bidding and targeting.

> **Design trade-off: Reporting attributed ROAS vs. leading with incremental ROAS.** Attributed ROAS is immediate, granular to the keyword and deterministic, but it overstates what the ads did, most for sponsored products shown to shoppers who were going to buy anyway. Incremental ROAS answers the question advertisers' finance teams ask, but it needs holdouts, is noisy for small campaigns and arrives later. Report attributed results from Stage 1 for day-to-day management, run periodic lift experiments for large advertisers at Stage 2, add always-on holdouts and iROAS reporting at Stage 3, and steer budget and autobidding decisions by incremental value as it becomes measurable (Chapters 9 and 14).

### 6.4. Closing the Loop

Measurement is useful only if it changes the next decision. The platform needs it to retrain models, recalibrate bids and detect fraud or reporting drift. Advertisers need it to reallocate budget, tighten or broaden targeting and retire weak creatives. It also tells them whether to keep the objective on traffic, shift to efficient sales or move up the funnel toward reach. The more automation the RMN introduces, the more this feedback path matters, because optimization systems learn from the labels the measurement stack produces.

---

## 7. Design Decisions and Trade-offs

This chapter's decisions concern the domain model and what the RMN offers advertisers. **Table 2.11** collects the trade-off callouts above.

**Table 2.11: Domain-Level Design Decisions for an RMN**

| Decision | Options | What each buys | When to choose |
|---|---|---|---|
| Entity model (§1.1) | One model for both formats vs. a separate display schema | One model: shared control plane, indexes, budgets and reporting; cross-format campaigns. Separate: simpler schemas | One model when display is built on the sponsored-products stack (Stage 3); separate only while display runs on a third-party ad server |
| Retrieval unit (§1.1) | Advertised products vs. SKUs | Advertised products: the right bid and budget reach the auction. SKUs: a smaller index | Index shared content per SKU and expand to advertised products before eligibility and bidding (Ch4) |
| Display buying model (§4.2) | Auction vs. guaranteed | Auction: per-impression pricing, no forecasting. Guaranteed: certainty for advertisers, predictable revenue | Guaranteed or sponsorship for premium placements once forecasting exists; auction for the rest |
| SP targeting (§4.3) | Manual keywords vs. auto-targeting | Manual: control and predictability. Auto: long-tail coverage with no setup | Both from Stage 1; auto-targeting as the default, manual keywords to refine |
| Headline metric (§6.3) | Attributed ROAS vs. incremental ROAS | Attributed: immediate and granular. Incremental: causal | Attributed from Stage 1; periodic lift tests at Stage 2; always-on holdouts and iROAS from Stage 3; incremental value for budget and bidding decisions |

---

## Summary and Next Steps

The domain model is the chapter's foundation. The shared hierarchy is Campaign → Ad Group → Ad. The sponsored-products retrieval unit is the advertised product, an (SKU, ad group) pair. The display delivery unit is the ads line item. Format, pricing model and buying model are separate axes, and targeting choices map those commercial decisions into the retrieval and eligibility machinery of Chapter 4.

The RMN in this book is one connected system: a control plane that turns campaign intent into serving state, a serving path that turns one request into a ranked, priced and placed ad, and a measurement loop that turns exposures and purchases back into budgets and reports. The **reference RMN** keeps those pieces numerically consistent: roughly 120 million monthly shoppers, 10 million advertised products, 15,000 average ad requests per second with headroom to 50,000 at peak, and a 60 ms search-integrated latency budget.

Measurement completes the picture. Attribution assigns observed outcomes to ads; incrementality asks whether the ads changed those outcomes at all. That distinction is the credibility boundary of the RMN business, and one reason this book treats measurement, ranking, bidding and page economics together.

Chapter 3 opens Part II with the serving architecture: it walks the request path in **Figure 2.2**, adds the display request flow and the control-plane mechanics, and turns this chapter's entities and controls into concrete services.

---

## References and Further Reading

[1] Walmart Inc. (2026). "Walmart reports Q4 results." Earnings release, February 19, 2026. https://stock.walmart.com/sec-filings/all-sec-filings/content/0000104169-26-000032/earningsreleasefy26q4.htm (Accessed October 2026). States that Walmart's "global advertising business grew 46% to nearly \$6.4 billion, including VIZIO" in fiscal 2026, and that Walmart Connect in the US grew 41% in the fourth quarter.

[2] Google Ad Manager Help. "Line item types and priorities." https://support.google.com/admanager/answer/177279 (Accessed October 2026). Describes guaranteed line items for directly sold campaigns: sponsorship line items deliver a defined percentage of impressions over a start and end time, and standard line items deliver against a defined impression goal over a start and end time.

[3] Ivie, G. (2015). "The Viewable Impression: The First Step To An Audience-Based Digital Currency." *MediaPost*, March 23, 2015. https://www.mediapost.com/publications/article/246125/the-viewable-impression-the-first-step-to-an-audi.html (Accessed October 2026). The Media Rating Council's chief executive describes the MRC viewable impression guidelines: 50% of an ad's pixels in view for at least one second for display, and for at least two seconds for video.

[4] Google Ads Help. "Formatting guidelines for uploading hashed data." Google Support. https://support.google.com/google-ads/answer/7476159 (Accessed October 2026). Gives the formatting rules for customer data before upload (email addresses lowercased with spaces removed, phone numbers in E.164 format) and states that advertisers can hash the data themselves with the SHA256 algorithm.

[5] Federal Trade Commission, Office of Technology. (2024). "No, hashing still doesn't make your data anonymous." *Technology Blog*, July 24, 2024. https://www.ftc.gov/policy/advocacy-research/tech-at-ftc/2024/07/no-hashing-still-doesnt-make-your-data-anonymous (Accessed October 2026). Describes hashing of email addresses, phone numbers and user IDs, and states that hashes are not anonymous and can still be used to identify users.

[6] Amazon Ads. "Sponsored Products." https://advertising.amazon.com/solutions/products/sponsored-products (Accessed October 2026). States that sponsored products ads may appear at the top of, alongside or within shopping results and on product pages, that advertisers bid the maximum they will pay when a shopper clicks, and that advertisers choose keywords or let Amazon target keywords automatically.

[7] Amazon Ads. "Your complete guide to Sponsored Brands." https://advertising.amazon.com/library/guides/sponsored-brands-what-to-know (Accessed October 2026). Describes Sponsored Brands as ads featuring a brand logo, headline, video or lifestyle image and multiple products, shown on top of and within shopping results, on product detail pages and on the homepage, priced by CPC or vCPM, with keyword or product targeting.

[8] Walmart. "Search results for coffee maker." https://www.walmart.com/search?q=coffee+maker (Accessed October 2026). Shows a sponsored Keurig brand unit above the coffee-maker results and sponsored product placements within the search experience.

[9] Walmart. "Mr. Coffee 5-Cup Programmable Coffee Maker." https://www.walmart.com/ip/Mr-Coffee-5-Cup-Programmable-Coffee-Maker/14017166848 (Accessed October 2026). Shows sponsored recommendation modules inside the product-detail page, including a sponsored carousel labeled "4 stars and above" and a separate sponsored offer module.

[10] Amazon Ads. "A guide to targeting with Sponsored Products." https://advertising.amazon.com/library/guides/targeting-with-sponsored-products (Accessed October 2026). Defines automatic targeting and its four groups (close match, loose match, substitutes, complements), manual keyword targeting with broad, phrase and exact match, negative keywords, and product targeting by products, categories, brands or features.

[11] Google Ads Help. "Keyword close variants: Definition." Google Support. https://support.google.com/google-ads/answer/9342105 (Accessed October 2026). Defines close variants (misspellings, singular and plural forms, stemming, abbreviations, accents, reordered words, function words, implied words, synonyms and paraphrases, same search intent) and states that all match types are eligible for them, with no way to opt out.

[12] Edelman, B., Ostrovsky, M., & Schwarz, M. (2007). "Internet Advertising and the Generalized Second-Price Auction: Selling Billions of Dollars Worth of Keywords." *American Economic Review*, 97(1), 242–259. https://www.aeaweb.org/articles?id=10.1257/aer.97.1.242 (Accessed October 2026). Shows that GSP generally has no dominant-strategy equilibrium and that truth-telling is not an equilibrium of GSP, unlike the VCG mechanism.

[13] Amazon Ads. "Guide to dynamic bidding: up and down with Sponsored Products." https://advertising.amazon.com/library/guides/dynamic-bidding-sponsored-products (Accessed October 2026). States that with dynamic bids up and down, Amazon increases or decreases bids by up to 100% based on performance, so a \$1.00 bid can rise to at most \$2.00.

[14] Google Ads Help. "About Target CPA bidding." Google Support. https://support.google.com/google-ads/answer/6268632 (Accessed October 2026). States that Target CPA bidding finds a bid each time an ad is eligible to appear, using historical campaign information and the contextual signals present at auction time.

[15] Google Ads Help. "About Target ROAS bidding." Google Support. https://support.google.com/google-ads/answer/6268637 (Accessed October 2026). States that Target ROAS predicts future conversions and their values and sets bids to maximize conversion value while trying to reach an average return on ad spend equal to the target.

[16] Blake, T., Nosko, C., & Tadelis, S. (2015). "Consumer Heterogeneity and Paid Search Effectiveness: A Large Scale Field Experiment." *Econometrica*, 83(1), 155–174. https://doi.org/10.3982/ECTA12423; working paper: https://www.nber.org/papers/w20171 (Accessed October 2026). Reports large-scale field experiments at eBay showing that returns from paid search are a fraction of conventional non-experimental estimates and that brand-keyword ads have no measurable short-term benefit.
