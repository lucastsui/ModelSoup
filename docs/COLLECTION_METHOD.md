# PropView data collection method (repeatable)

Recorded so collection can be re-run without rediscovering techniques.
Proven in multi-agent races (2026-08-09) and the London verification loop.

## When to use which track

| Track | Use for | Speed | Depth |
|-------|---------|-------|-------|
| **A. Rightmove HTTP parse** | Asking prices, live listings, FTB rent/mortgage charts, sold *page* summaries | Fast (~2–7 min fixed scope) | Regional asking often ~12 months; listings are snapshots |
| **B. Official bulk open data** | Long sold-price history (UKHPI), transaction grain (PPD), mortgage/Bank Rate time series | Fastest bulk (~2–4 min) | Decades of monthly sold HPI; PPD YTD; full BoE series |
| **C. Combined research pack** | Full research loop raw data | Track A + B + news | Price history + rates + news |

Research loop needs **historical price + political news**. Use **C**.

---

## Track A — Rightmove public pages (fastest equal-scope winner: agent_08)

### Proven settings
- **HTTP**: Python `urllib.request` (or `curl`)
- **User-Agent**: desktop Chrome string
- **Delay**: ~0.15–0.3 s between requests; retry once on failure
- **No Playwright** unless HTML is empty without JS (usually not required)
- **Save raw HTML** under `raw/` for audit/reparse

### Sources

| Dataset | URL |
|---------|-----|
| House Price Index (asking) | `https://www.rightmove.co.uk/news/house-price-index/` |
| Mortgage rate tracker | `https://www.rightmove.co.uk/news/articles/property-news/current-uk-mortgage-rates/` |
| Sold area + sample txs | `https://www.rightmove.co.uk/house-prices/{city_slug}.html` |
| For sale listings | `https://www.rightmove.co.uk/property-for-sale/{City}.html` |
| To rent listings | `https://www.rightmove.co.uk/property-to-rent/{City}.html` |

### Parse rules (do not re-invent)

1. **HPI national / MoM / stock / days-to-buyer / FTB series**  
   - Prefer HTML `<table>` rows (`<td>Month YYYY</td><td>value</td>`).  
   - Fallback: Chart.js `labels: [...]` + `data: [...]` near canvas blocks.

2. **HPI regional monthly**  
   - Parse hidden/data tables:  
     `Date | Region | £price | MoM% | YoY% | N days`  
   - Regions: Yorkshire and The Humber, West Midlands, East Midlands, East of England, North East, North West, South East, South West, London, Wales, Scotland.

3. **Market sectors (FTB / second-stepper / top)**  
   - Card text on HPI page; **do not** confuse with national headline £372k.  
   - Live targets (Jul 2026 vintage example): FTB £226,120; second-steppers £346,303; top £686,537.

4. **Mortgage tables**  
   - Parse each `<table>` in order:  
     (0) avg 2y/5y, (1) lowest 2y/5y, (2–4) avg by LTV, (5) **lowest FTB** by LTV.  
   - Label FTB table metrics `lowest_ftb_2y_fixed` / `lowest_ftb_5y_fixed` (not `avg_*`).

5. **Sold house-prices pages**  
   - Summary blurb: “overall average of £…”, type averages, YoY, vs peak.  
   - Cards: `data-testid="propertyCard"` + `<h2>` address + first dated `£` sale.

6. **Listings**  
   - Prefer `<script id="__NEXT_DATA__">` → walk for `properties` + `resultCount`.  
   - Fallback: split on `data-testid="propertyCard-N"` and CSS class price/address.  
   - Drop POA / absurd internal amounts (e.g. 65m display POA).

### Required output hygiene
- Every data row includes `source` and **`source_url`**.
- Maintain `DATA_SOURCE_REPORT.csv` (file → row_count → source_url).
- Optional independent verifier: re-fetch `source_url`, spot-check values → `VERIFICATION_STATUS.csv`.

### Reference implementations already on disk
- Equal-scope winner method notes: `agent_runs_equal/agent_08/METHOD.md`
- London verified pack: `london_verified/` + `COLLECTION_META.json`
- Root PropView CSVs from earlier Rightmove expands

---

## Track B — Official bulk open data (open-data race winner pattern: agent_07)

### Why
Long **sold-price** panels and **rate history** for backtests (MAE over many as-of dates). Rightmove asking HPI alone is too short regionally (~12 months).

### Sources (Open Government Licence / BoE public stats)

| Dataset | URL pattern |
|---------|-------------|
| UKHPI full file | `https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-YYYY-MM.csv` |
| Price Paid YTD | `https://price-paid-data.publicdata.landregistry.gov.uk/pp-YYYY.csv` |
| BoE Bank Rate | IADB series `IUDBEDR` |
| BoE quoted mortgages | e.g. `IUMBV34`, `IUMBV37`, `IUM2WTL`, `IUM5WTL`, … |

BoE CSV API shape (example):
```
https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2015&Dateto=now&SeriesCodes=IUDBEDR,IUMBV34,IUMBV37&UsingCodes=Y&VPD=Y&VFD=N
```

### Parse rules
1. **UKHPI**: filter RegionName / AreaCode to nations, English regions, major cities; keep `Date`, `AveragePrice`, `PercentageChange*`, `SalesVolume`, property-type columns if present.
2. **PPD**: no header in some files — standard 16-field Land Registry layout; aggregate by town/city + month.
3. **BoE**: long format `DATE, series, value` → monthly panels joinable to HPI on month.

### Reference
- `agent_runs/agent_07/METHOD.md` and outputs under `agent_runs/agent_07/`

---

## Track C — Full research pack (collector for the evolution loop)

### Raw data contract
```
raw_data = {
  historical_price_data,   # asking + sold + rent proxies
  rates_liquidity,         # mortgages, Bank Rate, days-to-buyer, stock
  political_news           # dated items usable as-of ≤ prediction as_of
}
```

### Output layout (this project)
```
research_data/
  COLLECTION_METHOD.md          # symlink or copy of this doc
  DATA_SOURCE_REPORT.csv
  COLLECTION_META.json
  raw/                          # downloaded HTML/CSV blobs
  prices/                       # tidy price series
  rates/                        # BoE + Rightmove mortgage snapshots
  news/                         # political/news events
  panels/                       # model-ready joins + prediction labels
  snapshots/                    # listings / sold samples (non-panel)
```

### Repeatable runner
```bash
python3 scripts/collect_full_research_data.py
```

### Prediction scoring grain (labels built from panels)
Each scored example needs actual future change % for:
- target: price | rent  
- aggregation: mean_asking | mean_sold | …  
- area, as_of, start, end  

Built from monthly panels with embargo: features use data ≤ as_of; label uses prices at start/end after as_of.

---

## Political / news collection method

| Source | URL / API | Fields |
|--------|-----------|--------|
| BBC Politics RSS | `https://feeds.bbci.co.uk/news/politics/rss.xml` | title, link, published, summary |
| Guardian Politics RSS | `https://www.theguardian.com/politics/rss` | same |
| BoE Bank Rate events | From BoE series diffs | date, event_type=rate_change, value |
| Optional: Rightmove HPI narrative | HPI article bullets | one-off macro context keys |

Store as `news/political_news_items.csv`:
`item_id, published_at, source, source_url, title, summary, tags`

Models may only use items with `published_at ≤ as_of_date`.

---

## Verification method (London loop)

1. Blind agent (no collection context) reads `DATA_SOURCE_REPORT.csv`.  
2. Re-fetches each `source_url`.  
3. Spot-checks ≥2–3 values per file; critical London headlines all checked.  
4. Writes `VERIFICATION_STATUS.csv` with PASS/FAIL/PARTIAL.  
5. Collector fixes FAILs; re-verify until all PASS.

Known past failures and fixes:
- Market sectors collapsed to national average → parse sector cards only / hard-check prices ≠ 372359.  
- Sold index count blank → parse results count or live page.  
- FTB mortgage rows mislabeled `avg_*` → use table order (table 5 = FTB lowest).  
- POA listings with internal 65m amount → drop from sample.

---

## Licence / terms notes
- Rightmove: public pages for research extracts; terms discourage bulk scraping — keep polite delays and minimal scope.  
- UKHPI / PPD: Open Government Licence.  
- BoE statistics: public with attribution as required.  
- News RSS: respect publisher terms; store URLs + titles/summaries for research.

---

## Changelog
- **2026-08-09**: Method recorded from agent races + London verification.  
- **2026-08-09**: Full research collector script added (`scripts/collect_full_research_data.py`).
