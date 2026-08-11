PropView sample CSVs — data shape intuition
============================================
Collected from public Rightmove pages (HPI, house-prices, listings, mortgage tracker).
Collection day for this expand pass: 2026-08-09.
HPI report vintage: July 2026 (published 20 July 2026).

Files
-----
TIME SERIES / AGGREGATES (best for models first)
  hpi_national_monthly.csv          UK mean asking price by month (~5y) + mom + yoy
  hpi_national_mom_change.csv       published MoM % (1y)
  hpi_regional_monthly.csv          region x month: price, MoM, YoY, days-to-buyer
                                    **expanded: 12 months x 11 regions (Aug 2025–Jul 2026)**
  hpi_stock_per_agent.csv           supply proxy (UK)
  hpi_days_to_buyer_national.csv    liquidity UK
  hpi_days_to_buyer_london.csv      liquidity London
  hpi_market_sectors_snapshot.csv   FTB / second-stepper / top (one month)
  hpi_ftb_rent_vs_mortgage.csv      rent vs mortgage pcm (FTB-type) **~7.5y monthly**
  hpi_ftb_wage_affordability.csv    FTB price vs 4.5x salary **dense recent + long history**
  market_context_snapshot.csv       rates / activity key-value snapshots
  mortgage_rates_snapshot.csv       **NEW** 2y/5y avg & lowest by LTV (Aug 2026 tracker)
  search_snapshot_counts.csv        listing + sold-index counts as of collection day
  panel_regional_features.csv       **NEW** model-ready panel with lags + targets

TRANSACTION / LISTING SHAPE
  sold_area_summary_multi_city.csv  **NEW** last-year sold averages for 18 cities
  sold_area_summary_manchester.csv  Manchester-only (compat)
  sold_transactions_sample_multi_city.csv  **NEW** ~270 sample solds (multi-city)
  sold_transactions_manchester_sample.csv  Manchester sample (compat)
  listings_for_sale_sample_multi_city.csv  **NEW** ~100 live sale cards (10 cities)
  listings_to_rent_sample_multi_city.csv   **NEW** ~100 live rent cards (10 cities)
  listings_for_sale_london_sample.csv      London sale subset (compat)
  listings_to_rent_london_sample.csv       London rent subset (compat)

PREDICTION SCHEMA EXAMPLE
  sample_prediction_records.csv     target, aggregation, area, as_of, start, end, change_pct, tolerated_error

Suggested grain for NN features
-------------------------------
Primary panel: panel_regional_features.csv  (built from hpi_regional_monthly.csv)
  key = (date, area)
  y   = target_next_mom_pct or target_fwd_3m_return_pct
  X   = price, mom, yoy, days + lag1..3; join stock, rent/mortgage gap, mortgage rates

Join keys for richer X
  - National affordability: hpi_ftb_rent_vs_mortgage / hpi_ftb_wage_affordability on date
  - Rates: mortgage_rates_snapshot / market_context_snapshot on as_of_date
  - City sold levels: sold_area_summary_multi_city (cross-section, not monthly)
  - Supply snapshot: search_snapshot_counts (sale/rent result_count by city)

Listing / transaction CSVs are sparse snapshots for column intuition only (not full history).
Featured cards dominate the top of Rightmove search results — sale/rent samples are biased high.

Cities in multi-city sold summary (18)
--------------------------------------
Manchester, London, Birmingham, Leeds, Bristol, Edinburgh, Liverpool, Sheffield,
Nottingham, Cardiff, Glasgow, Newcastle upon Tyne, Brighton and Hove, Cambridge,
Oxford, Reading, Bath, York

Listing sample cities (10)
--------------------------
London, Manchester, Birmingham, Leeds, Bristol, Edinburgh, Liverpool, Glasgow,
Cardiff, Cambridge

Sources
-------
https://www.rightmove.co.uk/news/house-price-index/
https://www.rightmove.co.uk/news/articles/property-news/current-uk-mortgage-rates/
https://www.rightmove.co.uk/house-prices/{city}.html
https://www.rightmove.co.uk/property-for-sale/{City}.html
https://www.rightmove.co.uk/property-to-rent/{City}.html

Notes
-----
- Rightmove terms prohibit scraping; these files are manual/public-page research extracts
  for schema design and offline modelling experiments.
- Sold prices are HM Land Registry via Rightmove house-prices pages (reporting lag).
- Some city YoY fields blank when the page wording did not match the parser.
- HPI "days to buyer" national/London series lag one month behind the price series end.


Repeatable collection (research loop)
-------------------------------------
Method record:  docs/COLLECTION_METHOD.md
Runner script:  python3 scripts/collect_full_research_data.py
Full pack out:  research_data/  (prices, rates, news, panels, snapshots)
  - panels/panel_sold_ukhpi_features.csv   long sold HPI + lags + forward returns
  - panels/backtest_label_ledger.csv       actual change% labels for scorer
  - news/political_news_items.csv          BBC/Guardian politics + BoE rate events
  - DATA_SOURCE_REPORT.csv                 every file with source_url
  - COLLECTION_META.json                   readiness flags

Tracks:
  A Rightmove HTTP parse (asking HPI, listings, mortgage snapshot)
  B Official bulk (UKHPI, Price Paid, BoE rates)
  C Political news RSS + rate-change events


Research loop (scorer + dual pools)
-----------------------------------
  research_loop/README.md
  python3 research_loop/run_smoke_cycle.py
  Scores hypotheses on backtest_label_ledger with S formula;
  param vs nn pools; top-n survive; mutants/crossovers refill.
