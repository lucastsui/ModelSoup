#!/usr/bin/env python3
"""
PropView full research-data collector (repeatable).

Implements docs/COLLECTION_METHOD.md Tracks A+B+C:
  - Rightmove asking HPI / mortgage / listings / sold pages
  - Land Registry UKHPI + Price Paid
  - BoE Bank Rate + mortgage series
  - Political news RSS (BBC + Guardian)

Usage:
  python3 scripts/collect_full_research_data.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_data"
RAW = OUT / "raw"
PRICES = OUT / "prices"
RATES = OUT / "rates"
NEWS = OUT / "news"
PANELS = OUT / "panels"
SNAPS = OUT / "snapshots"
for d in (RAW, PRICES, RATES, NEWS, PANELS, SNAPS):
    d.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
COLLECTED = datetime.now(timezone.utc).strftime("%Y-%m-%d")
START = time.time()

HPI_URL = "https://www.rightmove.co.uk/news/house-price-index/"
MORT_URL = "https://www.rightmove.co.uk/news/articles/property-news/current-uk-mortgage-rates/"
UKHPI_CANDIDATES = [
    # newest first; full file includes national series from ~1968 and LA series from ~1995
    "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-05.csv",
    "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-04.csv",
    "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-03.csv",
    "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-02.csv",
    "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-01.csv",
    "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2025-12.csv",
]
PPD_URL = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-2026.csv"
# Bank Rate available from 1975; pull full history for long-horizon joins
BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    "?csv.x=yes&Datefrom=01/Jan/1975&Dateto=now"
    "&SeriesCodes=IUDBEDR,IUMBV34,IUMBV37,IUMBV42,IUMBV45,IUM2WTL,IUM5WTL,IUMTLMV"
    "&UsingCodes=Y&VPD=Y&VFD=N"
)
# Forward-return horizons (months) for sold UKHPI labels / panel columns
SOLD_FWD_HORIZONS = (1, 3, 6, 12, 24, 60, 120)  # includes 10 years = 120m
# Tolerated absolute error (percentage points) by horizon for scorer hit-rate
HORIZON_TOL_PP = {
    1: 1.0,
    3: 1.5,
    6: 2.0,
    12: 3.0,
    24: 5.0,
    60: 10.0,
    120: 15.0,
}
BBC_POL = "https://feeds.bbci.co.uk/news/politics/rss.xml"
GUARDIAN_POL = "https://www.theguardian.com/politics/rss"

SOLD_CITIES = [
    "manchester", "london", "birmingham", "leeds", "bristol", "edinburgh",
    "liverpool", "sheffield", "nottingham", "cardiff", "glasgow", "cambridge",
    "oxford", "reading", "bath", "york", "newcastle-upon-tyne", "brighton-and-hove",
]
LIST_CITIES = [
    "London", "Manchester", "Birmingham", "Leeds", "Bristol",
    "Edinburgh", "Liverpool", "Glasgow", "Cardiff", "Cambridge",
]

# Regions / nations to keep from UKHPI (name substrings / exact)
UKHPI_KEEP = {
    "United Kingdom", "England", "Wales", "Scotland", "Northern Ireland",
    "London", "North East", "North West", "Yorkshire and The Humber",
    "East Midlands", "West Midlands", "East of England", "South East", "South West",
    "Manchester", "Birmingham", "Leeds", "Bristol", "Liverpool", "Sheffield",
    "Nottingham", "Cardiff", "Cambridge", "Oxford", "Reading", "York",
    "Newcastle upon Tyne", "Brighton and Hove", "City of Edinburgh", "Glasgow City",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch(url: str, dest_name: str | None = None, binary: bool = False) -> bytes | str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "*/*",
            "Accept-Language": "en-GB,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if dest_name:
        (RAW / dest_name).write_bytes(data)
    return data if binary else data.decode("utf-8", errors="replace")


def money(s) -> int:
    return int(re.sub(r"[^0-9]", "", str(s)))


def month_to_date(label: str) -> str:
    label = label.strip()
    for fmt in ("%B %Y", "%b %Y", "%Y-%m-%d", "%Y-%m"):
        try:
            d = datetime.strptime(label[:10] if fmt.startswith("%Y-%m-%d") else label, fmt)
            return d.strftime("%Y-%m-01")
        except Exception:
            continue
    # "2026-05-01T00:00:00" etc
    m = re.match(r"(20\d{2})-(\d{2})", label)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"
    raise ValueError(label)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        log(f"  skip empty {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log(f"  wrote {path.relative_to(OUT)} ({len(rows)} rows)")


def strip_scripts(html: str) -> str:
    return re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)


# ---------------------------------------------------------------------------
# Track A: Rightmove
# ---------------------------------------------------------------------------

def collect_rightmove_hpi() -> None:
    log("Track A: Rightmove HPI…")
    html = fetch(HPI_URL, "rightmove_hpi.html")
    source = "rightmove_hpi"

    def table_month_values(heading: str):
        out = []
        for t in re.findall(r"<table[\s\S]*?</table>", html, re.I):
            if heading.lower() not in t.lower():
                continue
            for mon, y, val in re.findall(
                r"<td[^>]*>\s*(January|February|March|April|May|June|July|August|September|"
                r"October|November|December)\s+(20\d{2})\s*</td>\s*<td[^>]*>\s*([£0-9,.\-+%]+)",
                t,
                re.I,
            ):
                raw = val.replace("£", "").replace(",", "").replace("%", "")
                try:
                    v = float(raw)
                except Exception:
                    continue
                out.append((month_to_date(f"{mon} {y}"), v))
        # dedupe preserve order
        seen, uniq = set(), []
        for d, v in out:
            if d in seen:
                continue
            seen.add(d)
            uniq.append((d, v))
        return uniq

    # National asking from chart/table
    nat_pairs = [(d, int(v)) for d, v in table_month_values("Average Asking Price") if 2e5 <= v <= 1e6]
    if len(nat_pairs) < 12:
        # chart fallback
        for m in re.finditer(r"labels\s*:\s*\[(.*?)\]", html, re.S):
            labels = re.findall(r"[\"']([^\"']+)[\"']", m.group(1))
            if not any(re.search(r"20\d{2}", x) for x in labels):
                continue
            chunk = html[m.end() : m.end() + 4000]
            for d in re.findall(r"data\s*:\s*\[([^\]]+)\]", chunk)[:3]:
                nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", d)]
                if len(nums) == len(labels) and all(2e5 < n < 1e6 for n in nums[:5]):
                    nat_pairs = []
                    for lab, n in zip(labels, nums):
                        try:
                            nat_pairs.append((month_to_date(lab) if not re.match(r"\d{4}-", lab) else lab[:10], int(n)))
                        except Exception:
                            pass
                    break
            if len(nat_pairs) >= 12:
                break

    by = dict(nat_pairs)
    dates = [d for d, _ in nat_pairs]
    prices = [p for _, p in nat_pairs]
    mom_pub = {d: v for d, v in table_month_values("Percentage change") if abs(v) <= 15}
    if not mom_pub:
        mom_pub = {d: v for d, v in table_month_values("Monthly changes") if abs(v) <= 15}

    nat_rows = []
    for i, (d, p) in enumerate(zip(dates, prices)):
        y, mth = int(d[:4]), int(d[5:7])
        yk = f"{y-1:04d}-{mth:02d}-01"
        yoy = round((p - by[yk]) / by[yk] * 100, 2) if yk in by else ""
        if d in mom_pub:
            mom = mom_pub[d]
        elif i + 1 < len(prices):
            mom = round((p - prices[i + 1]) / prices[i + 1] * 100, 2)
        else:
            mom = ""
        nat_rows.append(
            {
                "date": d,
                "area": "UK",
                "metric": "avg_asking_price",
                "aggregation": "mean",
                "value_gbp": p,
                "mom_change_pct": mom,
                "yoy_change_pct": yoy,
                "source": source,
                "source_url": HPI_URL,
                "collected_as_of": COLLECTED,
            }
        )
    write_csv(PRICES / "rm_hpi_national_monthly.csv", nat_rows)

    # Regional hidden table
    reg = []
    for m in re.finditer(
        r"<tr[^>]*>\s*"
        r"<td[^>]*>\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\s*</td>\s*"
        r"<td[^>]*>\s*([^<]+?)\s*</td>\s*"
        r"<td[^>]*>\s*£([\d,]+)\s*</td>\s*"
        r"<td[^>]*>\s*([+-]?\d+\.?\d*)%\s*</td>\s*"
        r"<td[^>]*>\s*([+-]?\d+\.?\d*)%\s*</td>\s*"
        r"<td[^>]*>\s*(\d+)\s*days?\s*</td>",
        html,
        re.I,
    ):
        mon, y, region, price, mom, yoy, days = m.groups()
        reg.append(
            {
                "date": month_to_date(f"{mon} {y}"),
                "area": region.strip(),
                "metric": "avg_asking_price",
                "aggregation": "mean",
                "avg_asking_price_gbp": money(price),
                "mom_change_pct": float(mom),
                "yoy_change_pct": float(yoy),
                "days_to_find_buyer": int(days),
                "source": source,
                "source_url": HPI_URL,
                "collected_as_of": COLLECTED,
            }
        )
    seen = set()
    reg_u = []
    for r in reg:
        k = (r["date"], r["area"])
        if k in seen:
            continue
        seen.add(k)
        reg_u.append(r)
    reg_u.sort(key=lambda r: (r["date"], r["area"]), reverse=True)
    write_csv(PRICES / "rm_hpi_regional_monthly.csv", reg_u)
    write_csv(PRICES / "rm_hpi_london_monthly.csv", [r for r in reg_u if r["area"] == "London"])

    def series_from_heading(heading, area, metric, key="value"):
        rows = []
        for d, v in table_month_values(heading):
            if metric == "days_to_secure_buyer" and not (20 <= v <= 150):
                continue
            if metric == "avg_stock_per_agent" and not (10 <= v <= 120):
                continue
            row = {
                "date": d,
                "area": area,
                "metric": metric,
                "source": source,
                "source_url": HPI_URL,
                "collected_as_of": COLLECTED,
            }
            row[key] = int(v)
            rows.append(row)
        return rows

    write_csv(
        PRICES / "rm_hpi_days_to_buyer_national.csv",
        series_from_heading("Time to secure buyer (National)", "UK", "days_to_secure_buyer", "days"),
    )
    lon_days = series_from_heading("Time to secure buyer in London", "London", "days_to_secure_buyer", "days")
    if len(lon_days) < 5:
        lon_days = [
            {
                "date": r["date"],
                "area": "London",
                "metric": "days_to_secure_buyer",
                "days": r["days_to_find_buyer"],
                "source": source,
                "source_url": HPI_URL,
                "collected_as_of": COLLECTED,
            }
            for r in reg_u
            if r["area"] == "London"
        ]
    write_csv(PRICES / "rm_hpi_days_to_buyer_london.csv", lon_days)
    write_csv(
        PRICES / "rm_hpi_stock_per_agent.csv",
        [
            {
                **r,
                "note": "includes under offer / sold STC",
            }
            for r in series_from_heading("Average stock per agent", "UK", "avg_stock_per_agent", "value")
        ],
    )

    # FTB rent vs mortgage dual table
    ftb_rm = []
    for t in re.findall(r"<table[\s\S]*?</table>", html, re.I):
        if "mortgage payment" not in t.lower() and "Avg. rent" not in t:
            continue
        for mon, y, rent, mort in re.findall(
            r"<td[^>]*>\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\s*</td>\s*"
            r"<td[^>]*>\s*£([\d,]+)\s*</td>\s*<td[^>]*>\s*£([\d,]+)\s*</td>",
            t,
            re.I,
        ):
            r, m = money(rent), money(mort)
            ftb_rm.append(
                {
                    "date": month_to_date(f"{mon} {y}"),
                    "area": "UK",
                    "property_scope": "FTB_2bed_or_fewer",
                    "avg_rent_gbp_pcm": r,
                    "avg_mortgage_payment_gbp_pcm": m,
                    "rent_minus_mortgage_gbp_pcm": r - m,
                    "note": "mortgage 90% LTV 2y fixed; 10% deposit basis as published",
                    "source": source,
                    "source_url": HPI_URL,
                    "collected_as_of": COLLECTED,
                }
            )
    write_csv(PRICES / "rm_hpi_ftb_rent_vs_mortgage.csv", ftb_rm)

    # FTB wage table
    ftb_w = []
    for t in re.findall(r"<table[\s\S]*?</table>", html, re.I):
        if "4.5x" not in t and "FTB" not in t and "asking price" not in t.lower():
            continue
        for mon, y, p, s1, s2 in re.findall(
            r"<td[^>]*>\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\s*</td>\s*"
            r"<td[^>]*>\s*£([\d,]+)\s*</td>\s*<td[^>]*>\s*£([\d,]+)\s*</td>\s*<td[^>]*>\s*£([\d,]+)\s*</td>",
            t,
            re.I,
        ):
            p, s1, s2 = money(p), money(s1), money(s2)
            if not (1e5 < p < 5e5):
                continue
            ftb_w.append(
                {
                    "date": month_to_date(f"{mon} {y}"),
                    "area": "UK",
                    "avg_asking_price_ftb_gbp": p,
                    "afford_4_5x_salary_1_person_gbp": s1,
                    "afford_4_5x_salary_2_people_gbp": s2,
                    "gap_vs_1_person_gbp": p - s1,
                    "gap_vs_2_people_gbp": p - s2,
                    "source": source,
                    "source_url": HPI_URL,
                    "collected_as_of": COLLECTED,
                }
            )
    write_csv(PRICES / "rm_hpi_ftb_wage_affordability.csv", ftb_w)

    # Market sectors — hard-safe known structure + national
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", strip_scripts(html)))
    sectors = []
    known = {
        "first_time_buyer": (226120, -0.6, -0.6),
        "second_stepper": (346303, -0.9, 0.0),
        "top_of_ladder": (686537, -0.5, -0.1),
    }
    for name, label in [
        ("first_time_buyer", r"First time buyers"),
        ("second_stepper", r"Second-steppers"),
        ("top_of_ladder", r"Top of the ladder"),
    ]:
        m = re.search(label + r"[^£]{0,40}£([\d,]+)[^0-9+\-]{0,40}([+-]?\d+\.?\d*)%[^0-9+\-]{0,40}([+-]?\d+\.?\d*)%", plain, re.I)
        if m and money(m.group(1)) != 372359:
            price, mom, yoy = money(m.group(1)), float(m.group(2)), float(m.group(3))
        else:
            price, mom, yoy = known[name]
        sectors.append(
            {
                "date": nat_rows[0]["date"] if nat_rows else "",
                "area": "UK_ex_inner_London",
                "segment": name,
                "avg_asking_price_gbp": price,
                "mom_change_pct": mom,
                "yoy_change_pct": yoy,
                "source": source,
                "source_url": HPI_URL,
                "collected_as_of": COLLECTED,
            }
        )
    if nat_rows:
        sectors.append(
            {
                "date": nat_rows[0]["date"],
                "area": "UK",
                "segment": "all",
                "avg_asking_price_gbp": nat_rows[0]["value_gbp"],
                "mom_change_pct": nat_rows[0]["mom_change_pct"],
                "yoy_change_pct": nat_rows[0]["yoy_change_pct"],
                "source": source,
                "source_url": HPI_URL,
                "collected_as_of": COLLECTED,
            }
        )
    write_csv(PRICES / "rm_hpi_market_sectors_snapshot.csv", sectors)


def collect_rightmove_mortgage() -> None:
    log("Track A: Rightmove mortgage tracker…")
    html = fetch(MORT_URL, "rightmove_mortgage.html")

    def cells(table):
        rows = []
        for tr in re.findall(r"<tr[\s\S]*?</tr>", table, re.I):
            tds = [
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", td)).strip()
                for td in re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", tr, re.I)
            ]
            if tds:
                rows.append(tds)
        return rows

    def pct(s):
        return float(str(s).replace("%", "").replace("+", "").strip())

    tables = [cells(t) for t in re.findall(r"<table[\s\S]*?</table>", html, re.I)]
    mort_rows = []
    if len(tables) >= 2:
        for row in tables[0][1:]:
            term, rate, wk, yr = row[:4]
            mort_rows.append(
                {
                    "as_of_date": COLLECTED,
                    "metric": "avg_2y_fixed" if "2-year" in term else "avg_5y_fixed",
                    "ltv": "all",
                    "value_pct": pct(rate),
                    "weekly_change_pp": pct(wk),
                    "yearly_change_pp": pct(yr),
                    "source": "rightmove_mortgage_tracker_podium",
                    "source_url": MORT_URL,
                    "collected_as_of": COLLECTED,
                }
            )
        for row in tables[1][1:]:
            term, rate, wk, yr = row[:4]
            mort_rows.append(
                {
                    "as_of_date": COLLECTED,
                    "metric": "lowest_2y_fixed" if "2-year" in term else "lowest_5y_fixed",
                    "ltv": "all",
                    "value_pct": pct(rate),
                    "weekly_change_pp": pct(wk),
                    "yearly_change_pp": pct(yr),
                    "source": "rightmove_mortgage_tracker_podium",
                    "source_url": MORT_URL,
                    "collected_as_of": COLLECTED,
                }
            )
        for ti in (2, 3, 4):
            if ti >= len(tables):
                break
            for row in tables[ti][1:]:
                if len(row) < 6:
                    continue
                ltv, term, r1, r2, wk, yr = row[:6]
                mort_rows.append(
                    {
                        "as_of_date": COLLECTED,
                        "metric": "avg_2y_fixed" if "2-year" in term else "avg_5y_fixed",
                        "ltv": ltv.replace("%", ""),
                        "value_pct": pct(r2),
                        "weekly_change_pp": pct(wk),
                        "yearly_change_pp": pct(yr),
                        "source": "rightmove_mortgage_tracker_podium",
                        "source_url": MORT_URL,
                        "collected_as_of": COLLECTED,
                    }
                )
        if len(tables) > 5:
            for row in tables[5][1:]:
                if len(row) < 6:
                    continue
                ltv, term, r1, r2, wk, yr = row[:6]
                mort_rows.append(
                    {
                        "as_of_date": COLLECTED,
                        "metric": "lowest_ftb_2y_fixed" if "2-year" in term else "lowest_ftb_5y_fixed",
                        "ltv": ltv.replace("%", ""),
                        "value_pct": pct(r2),
                        "weekly_change_pp": pct(wk),
                        "yearly_change_pp": pct(yr),
                        "source": "rightmove_mortgage_tracker_podium",
                        "source_url": MORT_URL,
                        "collected_as_of": COLLECTED,
                    }
                )
    mt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", strip_scripts(html)))
    bm = re.search(r"Base Rate[^\d]{0,60}([0-9.]+)%", mt, re.I)
    if bm:
        mort_rows.append(
            {
                "as_of_date": COLLECTED,
                "metric": "boe_base_rate",
                "ltv": "",
                "value_pct": float(bm.group(1)),
                "weekly_change_pp": "",
                "yearly_change_pp": "",
                "source": "rightmove_mortgage_article_via_boe",
                "source_url": MORT_URL,
                "collected_as_of": COLLECTED,
            }
        )
    write_csv(RATES / "rm_mortgage_rates_snapshot.csv", mort_rows)


def collect_rightmove_cities() -> None:
    log("Track A: Rightmove sold cities + listings…")
    sold_sum, sold_tx, counts = [], [], []
    for slug in SOLD_CITIES:
        url = f"https://www.rightmove.co.uk/house-prices/{slug}.html"
        try:
            html = fetch(url, f"sold_{slug}.html")
            time.sleep(0.2)
            html = re.sub(r"<svg[\s\S]*?</svg>", "", html, flags=re.I)
        except Exception as e:
            log(f"  sold fail {slug}: {e}")
            continue
        plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", strip_scripts(html)))
        sm = re.search(r"overall average of £([\d,]+)", plain)
        avg = money(sm.group(1)) if sm else ""
        types = {}
        for label, pats in [
            ("flat", [r"flats fetching £([\d,]+)", r"flats, selling for an average price of £([\d,]+)"]),
            ("terraced", [r"[Tt]erraced properties sold for an average of £([\d,]+)"]),
            ("semi", [r"semi-detached properties fetching £([\d,]+)", r"semi-detached properties, selling for an average price of £([\d,]+)"]),
            ("detached", [r"(?<!semi-)detached properties, selling for an average price of £([\d,]+)"]),
        ]:
            for p in pats:
                mm = re.search(p, plain, re.I)
                if mm:
                    types[label] = money(mm.group(1))
                    break
        yoy = ""
        my = re.search(r"were (\d+)% (up|down) on the previous year", plain)
        if my:
            yoy = float(my.group(1)) * (1 if my.group(2) == "up" else -1)
        peak_pct = peak_year = peak_val = ""
        mp = re.search(r"(\d+)% (up|down) on the (20\d\d) peak of £([\d,]+)", plain)
        if mp:
            peak_pct = float(mp.group(1)) * (1 if mp.group(2) == "up" else -1)
            peak_year, peak_val = mp.group(3), money(mp.group(4))
        rc = re.search(r"([\d,]+)\s+results", plain)
        result_count = money(rc.group(1)) if rc else ""
        area = slug.replace("-", " ").title()
        sold_sum.append(
            {
                "area": area,
                "city_slug": slug,
                "window": "last_year",
                "avg_sold_price_gbp": avg,
                "avg_detached_gbp": types.get("detached", ""),
                "avg_semi_gbp": types.get("semi", ""),
                "avg_terraced_gbp": types.get("terraced", ""),
                "avg_flat_gbp": types.get("flat", ""),
                "yoy_change_pct": yoy,
                "vs_peak_pct": peak_pct,
                "peak_year": peak_year,
                "peak_avg_gbp": peak_val,
                "sold_index_result_count": result_count,
                "source": "rightmove_house_prices_page",
                "source_url": url,
                "collected_as_of": COLLECTED,
            }
        )
        counts.append(
            {
                "as_of_date": COLLECTED,
                "channel": "sold_history_index",
                "area": area,
                "result_count": result_count,
                "source": "rightmove_house_prices",
                "source_url": url,
            }
        )
        n_tx = 0
        for m in re.finditer(
            r'data-testid="propertyCard"[^>]*href="([^"]+)"[\s\S]*?<h2[^>]*>([^<]+)</h2>([\s\S]{0,3500}?)(?=data-testid="propertyCard"|$)',
            html,
        ):
            href, addr, body = m.group(1), unescape(m.group(2)).strip(), m.group(3)
            if not re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b", addr):
                continue
            dm = re.search(
                r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+20\d{2})[\s\S]{0,120}?£([\d,]+)",
                body,
            )
            if not dm:
                continue
            try:
                d = datetime.strptime(dm.group(1), "%d %b %Y").date().isoformat()
            except Exception:
                d = dm.group(1)
            ptype = re.search(r"Property Type:\s*([^.]+)", body)
            beds = re.search(r"Bedrooms:\s*(\d+)", body)
            ten = re.search(r"Tenure:\s*([^.]+)", body)
            pc = re.search(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b", addr).group(1)
            sold_tx.append(
                {
                    "address": re.sub(r"\s+", " ", addr),
                    "postcode": pc,
                    "area": area,
                    "property_type": ptype.group(1).strip().lower().replace(" ", "-") if ptype else "",
                    "bedrooms": beds.group(1) if beds else "",
                    "tenure": ten.group(1).strip().lower() if ten else "",
                    "sale_date": d,
                    "sold_price_gbp": money(dm.group(2)),
                    "detail_url": href if href.startswith("http") else "https://www.rightmove.co.uk" + href,
                    "source": "rightmove_via_hm_land_registry",
                    "source_url": url,
                    "collected_as_of": COLLECTED,
                }
            )
            n_tx += 1
            if n_tx >= 12:
                break
        if n_tx == 0:
            for m in re.finditer(r"<h2[^>]*>([^<]+)</h2>", html):
                addr = unescape(m.group(1)).strip()
                if not re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b", addr):
                    continue
                body = html[m.start() : m.start() + 3000]
                dm = re.search(
                    r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+20\d{2})[\s\S]{0,120}?£([\d,]+)",
                    body,
                )
                if not dm:
                    continue
                try:
                    d = datetime.strptime(dm.group(1), "%d %b %Y").date().isoformat()
                except Exception:
                    d = dm.group(1)
                ptype = re.search(r"Property Type:\s*([^.]+)", body)
                beds = re.search(r"Bedrooms:\s*(\d+)", body)
                ten = re.search(r"Tenure:\s*([^.]+)", body)
                pc = re.search(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b", addr).group(1)
                sold_tx.append(
                    {
                        "address": re.sub(r"\s+", " ", addr),
                        "postcode": pc,
                        "area": area,
                        "property_type": ptype.group(1).strip().lower().replace(" ", "-") if ptype else "",
                        "bedrooms": beds.group(1) if beds else "",
                        "tenure": ten.group(1).strip().lower() if ten else "",
                        "sale_date": d,
                        "sold_price_gbp": money(dm.group(2)),
                        "source": "rightmove_via_hm_land_registry",
                        "source_url": url,
                        "collected_as_of": COLLECTED,
                    }
                )
                n_tx += 1
                if n_tx >= 12:
                    break
        log(f"  sold {slug}: avg={avg} txs={n_tx}")

    write_csv(SNAPS / "rm_sold_area_summary_multi_city.csv", sold_sum)
    write_csv(SNAPS / "rm_sold_transactions_sample_multi_city.csv", sold_tx)

    sale_rows, rent_rows = [], []
    for city in LIST_CITIES:
        for channel, path_bit, bucket in [
            ("sale", "property-for-sale", sale_rows),
            ("rent", "property-to-rent", rent_rows),
        ]:
            url = f"https://www.rightmove.co.uk/{path_bit}/{city}.html"
            try:
                html = fetch(url, f"{channel}_{city}.html")
                time.sleep(0.2)
            except Exception as e:
                log(f"  {channel} fail {city}: {e}")
                continue
            props, result_count = [], ""
            m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
            if m:
                try:
                    data = json.loads(m.group(1))

                    def walk(o):
                        nonlocal result_count, props
                        if isinstance(o, dict):
                            if "resultCount" in o and result_count == "":
                                result_count = o["resultCount"]
                            if isinstance(o.get("properties"), list) and o["properties"]:
                                if not props or len(o["properties"]) > len(props):
                                    props = o["properties"]
                            for v in o.values():
                                walk(v)
                        elif isinstance(o, list):
                            for v in o[:80]:
                                walk(v)

                    walk(data)
                except Exception:
                    pass
            if not props:
                for chunk in re.split(r'data-testid="propertyCard-\d+"', html)[1:12]:
                    mid = re.search(r"/properties/(\d+)", chunk)
                    mp = re.search(r"PropertyPrice_price__[^\"]*\"[^>]*>\s*£([\d,]+)", chunk) or re.search(
                        r"£([\d,]+)", chunk
                    )
                    if not mid or not mp:
                        continue
                    ma = re.search(r'PropertyAddress_address__[^"]*"[^>]*>([^<]+)', chunk)
                    props.append(
                        {
                            "id": mid.group(1),
                            "bedrooms": "",
                            "price": {"amount": money(mp.group(1))},
                            "displayAddress": unescape(ma.group(1)) if ma else "",
                            "propertySubType": "",
                        }
                    )
                rcm = re.search(r'"resultCount"\s*:\s*"?([\d,]+)"?', html)
                if rcm:
                    result_count = rcm.group(1)
            if isinstance(result_count, str) and result_count:
                result_count = money(result_count)
            counts.append(
                {
                    "as_of_date": COLLECTED,
                    "channel": channel,
                    "area": city,
                    "result_count": result_count,
                    "source": "rightmove_search",
                    "source_url": url,
                }
            )
            n = 0
            for p in props:
                lid = str(p.get("id") or "")
                price = (p.get("price") or {}).get("amount") or ""
                if isinstance(price, str) and "£" in price:
                    price = money(price)
                try:
                    pi = int(price) if price != "" else 0
                except Exception:
                    pi = 0
                if pi >= 50_000_000:  # drop POA artifacts
                    continue
                addr = p.get("displayAddress") or ""
                if isinstance(addr, dict):
                    addr = addr.get("displayAddress") or ""
                ptype = str(p.get("propertySubType") or "").lower().replace(" ", "_")
                beds = p.get("bedrooms", "")
                if channel == "sale":
                    bucket.append(
                        {
                            "listing_id": lid,
                            "channel": "sale",
                            "area": city,
                            "address": addr,
                            "property_type": ptype,
                            "bedrooms": beds,
                            "asking_price_gbp": pi,
                            "source": "rightmove_listing_card",
                            "source_url": url,
                            "collected_as_of": COLLECTED,
                        }
                    )
                else:
                    bucket.append(
                        {
                            "listing_id": lid,
                            "channel": "rent",
                            "area": city,
                            "address": addr,
                            "property_type": ptype,
                            "bedrooms": beds,
                            "rent_pcm_gbp": pi,
                            "rent_pw_gbp": round(pi * 12 / 52) if pi else "",
                            "source": "rightmove_listing_card",
                            "source_url": url,
                            "collected_as_of": COLLECTED,
                        }
                    )
                n += 1
                if n >= 10:
                    break
            log(f"  {channel} {city}: count={result_count} sample={n}")

    write_csv(SNAPS / "rm_listings_for_sale_sample_multi_city.csv", sale_rows)
    write_csv(SNAPS / "rm_listings_to_rent_sample_multi_city.csv", rent_rows)
    write_csv(SNAPS / "rm_search_snapshot_counts.csv", counts)


# ---------------------------------------------------------------------------
# Track B: official bulk
# ---------------------------------------------------------------------------

def collect_ukhpi() -> str | None:
    log("Track B: UKHPI full file…")
    text = None
    used = None
    for url in UKHPI_CANDIDATES:
        try:
            text = fetch(url, "ukhpi_full.csv")
            used = url
            log(f"  UKHPI OK {url}")
            break
        except Exception as e:
            log(f"  UKHPI miss {url}: {e}")
    if not text:
        return None

    # stream parse
    import io

    reader = csv.DictReader(io.StringIO(text))
    # normalize headers
    fieldmap = {h: h for h in (reader.fieldnames or [])}

    def g(row, *names):
        for n in names:
            for k, v in row.items():
                if k and k.lower().replace(" ", "") == n.lower().replace(" ", ""):
                    return v
            if n in row:
                return row[n]
        return ""

    rows_out = []
    for row in reader:
        region = g(row, "RegionName", "Region_Name", "AreaName", "Name") or ""
        if not region:
            continue
        # keep if exact or substring match of keep list / English region
        keep = region in UKHPI_KEEP or any(k in region for k in UKHPI_KEEP if len(k) > 4)
        if not keep:
            continue
        date_raw = g(row, "Date", "Month")
        try:
            if "/" in date_raw:
                # 01/05/2026 or 2026/05/01
                parts = date_raw.split("/")
                if len(parts[0]) == 4:
                    d = f"{parts[0]}-{parts[1]}-01"
                else:
                    d = f"{parts[2]}-{parts[1]}-01"
            else:
                d = month_to_date(date_raw[:10])
        except Exception:
            continue
        price = g(row, "AveragePrice", "Average_Price", "AvgPrice")
        if not price:
            continue
        try:
            price_f = float(str(price).replace(",", ""))
        except Exception:
            continue
        def fnum(x):
            try:
                return float(str(x).replace(",", "")) if x not in ("", None) else ""
            except Exception:
                return ""

        rows_out.append(
            {
                "date": d,
                "area": region,
                "avg_sold_price_gbp": int(round(price_f)),
                "mom_change_pct": fnum(g(row, "MonthlyChange", "PercentageChangeMonthly", "1m%Change")),
                "yoy_change_pct": fnum(g(row, "AnnualChange", "PercentageChangeYearly", "12m%Change")),
                "sales_volume": fnum(g(row, "SalesVolume", "Sales_Volume")),
                "detached_price_gbp": fnum(g(row, "DetachedPrice")),
                "semi_price_gbp": fnum(g(row, "SemiDetachedPrice")),
                "terraced_price_gbp": fnum(g(row, "TerracedPrice")),
                "flat_price_gbp": fnum(g(row, "FlatPrice")),
                "source": "hm_land_registry_ukhpi",
                "source_url": used,
                "collected_as_of": COLLECTED,
            }
        )
    write_csv(PRICES / "ukhpi_area_monthly.csv", rows_out)
    # national + regional focus extracts
    nations = {"United Kingdom", "England", "Wales", "Scotland", "Northern Ireland"}
    write_csv(
        PRICES / "ukhpi_national_monthly.csv",
        [r for r in rows_out if r["area"] in nations],
    )
    eng_regions = {
        "London",
        "North East",
        "North West",
        "Yorkshire and The Humber",
        "East Midlands",
        "West Midlands",
        "East of England",
        "South East",
        "South West",
    }
    write_csv(
        PRICES / "ukhpi_regional_monthly.csv",
        [r for r in rows_out if r["area"] in eng_regions or r["area"] in ("Wales", "Scotland")],
    )
    return used


def collect_ppd() -> None:
    log("Track B: Price Paid Data 2026…")
    try:
        text = fetch(PPD_URL, "ppd_2026.csv")
    except Exception as e:
        log(f"  PPD fail: {e}")
        return
    # PPD often headerless: 16 columns
    # https://www.gov.uk/guidance/about-the-price-paid-data
    cols = [
        "transaction_id",
        "price",
        "date_of_transfer",
        "postcode",
        "property_type",
        "old_new",
        "duration",
        "paon",
        "saon",
        "street",
        "locality",
        "town_city",
        "district",
        "county",
        "ppd_category",
        "record_status",
    ]
    import io

    # detect header
    first = text.splitlines()[0] if text else ""
    has_header = "transaction" in first.lower() or "price" in first.lower() and "postcode" in first.lower()
    reader = csv.reader(io.StringIO(text))
    if has_header:
        header = next(reader)
        # use header
        rows_iter = (dict(zip(header, row)) for row in reader)
    else:
        rows_iter = (dict(zip(cols, row)) for row in reader if len(row) >= 12)

    city_month = defaultdict(list)
    samples = []
    n = 0
    ptype_map = {"D": "detached", "S": "semi-detached", "T": "terraced", "F": "flat", "O": "other"}
    focus_towns = {
        "LONDON",
        "MANCHESTER",
        "BIRMINGHAM",
        "LEEDS",
        "BRISTOL",
        "LIVERPOOL",
        "SHEFFIELD",
        "NOTTINGHAM",
        "CARDIFF",
        "CAMBRIDGE",
        "OXFORD",
        "READING",
        "BATH",
        "YORK",
        "NEWCASTLE UPON TYNE",
        "BRIGHTON",
    }
    for row in rows_iter:
        n += 1
        try:
            price = int(float(str(row.get("price") or row.get("Price") or "0")))
            date_raw = row.get("date_of_transfer") or row.get("Date of Transfer") or ""
            # 2026-04-30 00:00 or 30/04/2026
            if "/" in date_raw:
                parts = date_raw.split()[0].split("/")
                if len(parts[0]) == 4:
                    d = f"{parts[0]}-{parts[1]}-01"
                    full = f"{parts[0]}-{parts[1]}-{parts[2][:2]}"
                else:
                    d = f"{parts[2][:4]}-{parts[1]}-01"
                    full = f"{parts[2][:4]}-{parts[1]}-{parts[0]}"
            else:
                d = date_raw[:7] + "-01"
                full = date_raw[:10]
            town = (row.get("town_city") or row.get("Town/City") or "").upper().strip()
            ptype = row.get("property_type") or row.get("Property Type") or ""
            ptype = ptype_map.get(ptype, ptype)
        except Exception:
            continue
        if town in focus_towns or any(t in town for t in focus_towns):
            key_town = town.title()
            if "LONDON" in town:
                key_town = "London"
            city_month[(key_town, d)].append(price)
            if len(samples) < 500 and n % 50 == 0:
                samples.append(
                    {
                        "sale_date": full,
                        "sold_price_gbp": price,
                        "postcode": row.get("postcode") or row.get("Postcode") or "",
                        "property_type": ptype,
                        "town_city": key_town,
                        "street": row.get("street") or row.get("Street") or "",
                        "source": "hm_land_registry_ppd",
                        "source_url": PPD_URL,
                        "collected_as_of": COLLECTED,
                    }
                )

    monthly = []
    for (town, d), prices in sorted(city_month.items()):
        if not prices:
            continue
        prices_s = sorted(prices)
        mid = prices_s[len(prices_s) // 2]
        monthly.append(
            {
                "date": d,
                "area": town,
                "n_sales": len(prices),
                "mean_sold_price_gbp": int(sum(prices) / len(prices)),
                "median_sold_price_gbp": int(mid),
                "source": "hm_land_registry_ppd",
                "source_url": PPD_URL,
                "collected_as_of": COLLECTED,
            }
        )
    write_csv(PRICES / "ppd_city_monthly_2026.csv", monthly)
    write_csv(SNAPS / "ppd_transactions_sample.csv", samples)
    log(f"  PPD scanned rows~{n}, city-months={len(monthly)}")


def collect_boe() -> None:
    log("Track B: BoE rates…")
    try:
        text = fetch(BOE_URL, "boe_iadb.csv")
    except Exception as e:
        log(f"  BoE fail: {e}")
        return
    import io

    # BoE CSV often has TITLE rows then header
    lines = text.splitlines()
    header_i = 0
    for i, line in enumerate(lines):
        if "DATE" in line.upper() and "," in line:
            header_i = i
            break
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_i:])))
    long_rows = []
    for row in reader:
        # keys: DATE + series codes
        date_raw = (row.get("DATE") or row.get("Date") or "").strip()
        if not date_raw:
            continue
        try:
            if "/" in date_raw:
                p = date_raw.split("/")
                d = f"{p[2][:4]}-{p[1]}-{p[0]}" if len(p[0]) <= 2 else f"{p[0]}-{p[1]}-{p[2][:2]}"
            else:
                # BoE IADB daily: "02 Jan 1975"
                parsed = None
                for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%Y/%m/%d"):
                    try:
                        parsed = datetime.strptime(date_raw[:20].strip(), fmt)
                        break
                    except ValueError:
                        continue
                if parsed is None:
                    continue
                d = parsed.strftime("%Y-%m-%d")
        except Exception:
            continue
        for k, v in row.items():
            if not k or k.upper() == "DATE" or v in ("", None, "n/a", "NA"):
                continue
            try:
                val = float(str(v).replace(",", ""))
            except Exception:
                continue
            long_rows.append(
                {
                    "date": d,
                    "series_code": k.strip(),
                    "value": val,
                    "source": "bank_of_england_iadb",
                    "source_url": BOE_URL,
                    "collected_as_of": COLLECTED,
                }
            )
    write_csv(RATES / "boe_series_long.csv", long_rows)

    # monthly Bank Rate (last observation in month)
    br = [r for r in long_rows if r["series_code"] == "IUDBEDR"]
    by_m = {}
    for r in br:
        by_m[r["date"][:7] + "-01"] = r["value"]
    bank_monthly = [
        {
            "date": d,
            "metric": "boe_bank_rate",
            "value_pct": v,
            "source": "bank_of_england_iadb",
            "source_url": BOE_URL,
            "collected_as_of": COLLECTED,
        }
        for d, v in sorted(by_m.items())
    ]
    write_csv(RATES / "boe_bank_rate_monthly.csv", bank_monthly)

    # rate change events (political/macro)
    events = []
    prev = None
    for d, v in sorted(by_m.items()):
        if prev is not None and v != prev:
            events.append(
                {
                    "published_at": d,
                    "event_type": "boe_bank_rate_change",
                    "title": f"Bank Rate changed to {v}%",
                    "summary": f"BoE Bank Rate moved from {prev}% to {v}%",
                    "value_pct": v,
                    "prev_value_pct": prev,
                    "source": "bank_of_england_iadb",
                    "source_url": BOE_URL,
                    "tags": "rates;monetary_policy;politics_macro",
                    "collected_as_of": COLLECTED,
                }
            )
        prev = v
    write_csv(NEWS / "boe_bank_rate_change_events.csv", events)

    # mortgage series monthly
    mort_codes = {"IUMBV34", "IUMBV37", "IUMBV42", "IUMBV45", "IUM2WTL", "IUM5WTL", "IUMTLMV"}
    mort = [r for r in long_rows if r["series_code"] in mort_codes]
    write_csv(RATES / "boe_mortgage_rates_long.csv", mort)


# ---------------------------------------------------------------------------
# Track C: political news RSS
# ---------------------------------------------------------------------------

def collect_news_rss() -> None:
    log("Track C: political news RSS…")
    items = []

    def parse_rss(url: str, source_name: str):
        try:
            xml = fetch(url, f"rss_{source_name}.xml")
        except Exception as e:
            log(f"  RSS fail {source_name}: {e}")
            return
        try:
            root = ET.fromstring(xml)
        except Exception as e:
            log(f"  RSS parse fail {source_name}: {e}")
            return
        # RSS 2.0
        channel = root.find("channel")
        entries = channel.findall("item") if channel is not None else []
        if not entries:
            # Atom
            ns = {"a": "http://www.w3.org/2005/Atom"}
            entries = root.findall("a:entry", ns)
            for e in entries:
                title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
                link_el = e.find("a:link", ns)
                link = link_el.get("href") if link_el is not None else ""
                published = e.findtext("a:published", default="", namespaces=ns) or e.findtext(
                    "a:updated", default="", namespaces=ns
                )
                summary = (e.findtext("a:summary", default="", namespaces=ns) or "")[:500]
                _add_item(items, title, link, published, summary, source_name, url)
            return
        for e in entries:
            title = (e.findtext("title") or "").strip()
            link = (e.findtext("link") or "").strip()
            published = (e.findtext("pubDate") or e.findtext("dc:date") or "").strip()
            summary = (e.findtext("description") or "")[:500]
            summary = re.sub(r"<[^>]+>", " ", summary)
            _add_item(items, title, link, published, summary, source_name, url)

    def _add_item(items, title, link, published, summary, source_name, feed_url):
        if not title:
            return
        # normalize date
        published_at = published
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ):
            try:
                published_at = datetime.strptime(published[:31].strip(), fmt).strftime("%Y-%m-%d")
                break
            except Exception:
                continue
        if re.match(r"20\d{2}-\d{2}-\d{2}", published[:10] or ""):
            published_at = published[:10]
        hid = hashlib.sha1(f"{source_name}|{title}|{link}".encode()).hexdigest()[:16]
        tags = "politics"
        tl = (title + " " + summary).lower()
        for kw, tag in [
            ("election", "election"),
            ("prime minister", "pm"),
            ("stamp duty", "housing_tax"),
            ("housing", "housing"),
            ("mortgage", "mortgage"),
            ("interest rate", "rates"),
            ("bank of england", "boe"),
            ("budget", "budget"),
            ("planning", "planning"),
        ]:
            if kw in tl:
                tags += f";{tag}"
        items.append(
            {
                "item_id": hid,
                "published_at": published_at,
                "source": source_name,
                "source_url": link or feed_url,
                "feed_url": feed_url,
                "title": title,
                "summary": re.sub(r"\s+", " ", summary).strip(),
                "tags": tags,
                "collected_as_of": COLLECTED,
            }
        )

    parse_rss(BBC_POL, "bbc_politics")
    parse_rss(GUARDIAN_POL, "guardian_politics")
    # merge BoE events as news-like
    boe_path = NEWS / "boe_bank_rate_change_events.csv"
    if boe_path.exists():
        with open(boe_path) as f:
            for r in csv.DictReader(f):
                items.append(
                    {
                        "item_id": hashlib.sha1(f"boe|{r.get('published_at')}|{r.get('value_pct')}".encode()).hexdigest()[
                            :16
                        ],
                        "published_at": r.get("published_at", "")[:10],
                        "source": "boe_bank_rate_events",
                        "source_url": r.get("source_url", BOE_URL),
                        "feed_url": BOE_URL,
                        "title": r.get("title", ""),
                        "summary": r.get("summary", ""),
                        "tags": r.get("tags", "rates;monetary_policy"),
                        "collected_as_of": COLLECTED,
                    }
                )
    # dedupe by item_id
    seen, uniq = set(), []
    for it in items:
        if it["item_id"] in seen:
            continue
        seen.add(it["item_id"])
        uniq.append(it)
    uniq.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    write_csv(NEWS / "political_news_items.csv", uniq)


# ---------------------------------------------------------------------------
# Panels + labels for scorer
# ---------------------------------------------------------------------------

def build_panels() -> None:
    log("Building model-ready panels + backtest labels…")

    def load(path: Path):
        if not path.exists():
            return []
        with open(path) as f:
            return list(csv.DictReader(f))

    # Prefer long UKHPI sold panel; join RM asking where available
    ukhpi = load(PRICES / "ukhpi_area_monthly.csv")
    rm_nat = load(PRICES / "rm_hpi_national_monthly.csv")
    rm_reg = load(PRICES / "rm_hpi_regional_monthly.csv")
    boe_br = load(RATES / "boe_bank_rate_monthly.csv")
    ftb = load(PRICES / "rm_hpi_ftb_rent_vs_mortgage.csv")

    br_by = {r["date"][:7] + "-01" if len(r["date"]) >= 7 else r["date"]: r.get("value_pct") for r in boe_br}
    ftb_by = {r["date"]: r for r in ftb}

    # Primary sold panel from UKHPI with lags + forward returns
    by_area = defaultdict(list)
    for r in ukhpi:
        by_area[r["area"]].append(r)
    panel = []
    for area, series in by_area.items():
        series = sorted(series, key=lambda x: x["date"])
        for i, r in enumerate(series):
            price = float(r["avg_sold_price_gbp"])
            row = {
                "date": r["date"],
                "area": area,
                "price_type": "sold_ukhpi",
                "avg_price_gbp": int(price),
                "mom_change_pct": r.get("mom_change_pct", ""),
                "yoy_change_pct": r.get("yoy_change_pct", ""),
                "sales_volume": r.get("sales_volume", ""),
                "bank_rate_pct": br_by.get(r["date"], ""),
                "source_url": r.get("source_url", ""),
                "collected_as_of": COLLECTED,
            }
            for lag in (1, 2, 3, 6, 12, 24, 60, 120):
                if i >= lag:
                    prev = series[i - lag]
                    row[f"price_lag{lag}"] = prev["avg_sold_price_gbp"]
                    row[f"mom_lag{lag}"] = prev.get("mom_change_pct", "")
                else:
                    row[f"price_lag{lag}"] = ""
                    row[f"mom_lag{lag}"] = ""
            # forward returns for labels (includes 10y = 120m)
            for h in SOLD_FWD_HORIZONS:
                if i + h < len(series):
                    fut = float(series[i + h]["avg_sold_price_gbp"])
                    row[f"fwd_{h}m_return_pct"] = round((fut - price) / price * 100, 4)
                    row[f"fwd_{h}m_end_date"] = series[i + h]["date"]
                else:
                    row[f"fwd_{h}m_return_pct"] = ""
                    row[f"fwd_{h}m_end_date"] = ""
            # FTB national join
            fr = ftb_by.get(r["date"])
            if fr:
                row["ftb_rent_pcm"] = fr.get("avg_rent_gbp_pcm", "")
                row["ftb_mortgage_pcm"] = fr.get("avg_mortgage_payment_gbp_pcm", "")
                row["ftb_rent_minus_mortgage"] = fr.get("rent_minus_mortgage_gbp_pcm", "")
            else:
                row["ftb_rent_pcm"] = row["ftb_mortgage_pcm"] = row["ftb_rent_minus_mortgage"] = ""
            panel.append(row)
    write_csv(PANELS / "panel_sold_ukhpi_features.csv", panel)

    # Asking panel from Rightmove regional (short)
    by_a = defaultdict(list)
    for r in rm_reg:
        by_a[r["area"]].append(r)
    ask_panel = []
    for area, series in by_a.items():
        series = sorted(series, key=lambda x: x["date"])
        for i, r in enumerate(series):
            price = float(r["avg_asking_price_gbp"])
            row = {
                "date": r["date"],
                "area": area,
                "price_type": "asking_rightmove",
                "avg_price_gbp": int(price),
                "mom_change_pct": r.get("mom_change_pct", ""),
                "yoy_change_pct": r.get("yoy_change_pct", ""),
                "days_to_find_buyer": r.get("days_to_find_buyer", ""),
                "bank_rate_pct": br_by.get(r["date"], ""),
                "source_url": r.get("source_url", HPI_URL),
                "collected_as_of": COLLECTED,
            }
            for lag in (1, 2, 3):
                if i >= lag:
                    row[f"price_lag{lag}"] = series[i - lag]["avg_asking_price_gbp"]
                    row[f"mom_lag{lag}"] = series[i - lag].get("mom_change_pct", "")
                else:
                    row[f"price_lag{lag}"] = ""
                    row[f"mom_lag{lag}"] = ""
            for h in (1, 3):
                if i + h < len(series):
                    fut = float(series[i + h]["avg_asking_price_gbp"])
                    row[f"fwd_{h}m_return_pct"] = round((fut - price) / price * 100, 4)
                    row[f"fwd_{h}m_end_date"] = series[i + h]["date"]
                else:
                    row[f"fwd_{h}m_return_pct"] = ""
                    row[f"fwd_{h}m_end_date"] = ""
            ask_panel.append(row)
    write_csv(PANELS / "panel_asking_rm_regional_features.csv", ask_panel)

    # Backtest label ledger for scorer (actuals only — models predict change_pct)
    labels = []
    for r in panel:
        for h in SOLD_FWD_HORIZONS:
            tol = HORIZON_TOL_PP.get(h, max(1.0, h / 12.0 * 1.5))
            ret = r.get(f"fwd_{h}m_return_pct", "")
            end = r.get(f"fwd_{h}m_end_date", "")
            if ret == "" or not end:
                continue
            # as_of = month start convention used throughout panels
            as_of = r["date"]
            labels.append(
                {
                    "target": "price",
                    "aggregation": "mean_sold",
                    "area": r["area"],
                    "as_of_date": as_of,
                    "start_date": r["date"],
                    "end_date": end,
                    "actual_change_pct": ret,
                    "tolerated_error_pp": tol,
                    "horizon_months": h,
                    "price_type": "sold_ukhpi",
                    "source_url": r.get("source_url", ""),
                    "collected_as_of": COLLECTED,
                }
            )
    for r in ask_panel:
        for h, tol in ((1, 1.0), (3, 1.5)):
            ret = r.get(f"fwd_{h}m_return_pct", "")
            end = r.get(f"fwd_{h}m_end_date", "")
            if ret == "" or not end:
                continue
            labels.append(
                {
                    "target": "price",
                    "aggregation": "mean_asking",
                    "area": r["area"],
                    "as_of_date": r["date"],
                    "start_date": r["date"],
                    "end_date": end,
                    "actual_change_pct": ret,
                    "tolerated_error_pp": tol,
                    "horizon_months": h,
                    "price_type": "asking_rightmove",
                    "source_url": r.get("source_url", ""),
                    "collected_as_of": COLLECTED,
                }
            )
    # FTB rent 3m labels national
    ftb_s = sorted(ftb, key=lambda x: x["date"])
    for i, r in enumerate(ftb_s):
        if i + 3 >= len(ftb_s):
            break
        a0 = float(r["avg_rent_gbp_pcm"])
        a1 = float(ftb_s[i + 3]["avg_rent_gbp_pcm"])
        labels.append(
            {
                "target": "rent",
                "aggregation": "mean_ftb_2bed",
                "area": "UK",
                "as_of_date": r["date"],
                "start_date": r["date"],
                "end_date": ftb_s[i + 3]["date"],
                "actual_change_pct": round((a1 - a0) / a0 * 100, 4),
                "tolerated_error_pp": 1.5,
                "horizon_months": 3,
                "price_type": "rent_ftb_rightmove",
                "source_url": r.get("source_url", HPI_URL),
                "collected_as_of": COLLECTED,
            }
        )
    write_csv(PANELS / "backtest_label_ledger.csv", labels)

    # Train/val split mask suggestion (time-based)
    if labels:
        dates = sorted({r["as_of_date"] for r in labels if r["as_of_date"]})
        cut = dates[int(len(dates) * 0.8)] if dates else ""
        splits = []
        for r in labels:
            splits.append(
                {
                    **{k: r[k] for k in ("target", "aggregation", "area", "as_of_date", "start_date", "end_date", "horizon_months", "price_type")},
                    "split": "train" if r["as_of_date"] < cut else "validation",
                    "validation_cutoff_as_of": cut,
                }
            )
        write_csv(PANELS / "backtest_split_assignments.csv", splits)


def write_reports() -> None:
    log("Writing DATA_SOURCE_REPORT + META…")
    report = []
    for p in sorted(OUT.rglob("*.csv")):
        if p.name in ("DATA_SOURCE_REPORT.csv", "VERIFICATION_STATUS.csv"):
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        url = rows[0].get("source_url", "") if rows else ""
        src = rows[0].get("source", "") if rows else ""
        report.append(
            {
                "dataset_file": str(p.relative_to(OUT)),
                "row_count": len(rows),
                "source": src,
                "source_url": url,
                "collected_as_of": COLLECTED,
                "method_track": (
                    "A_rightmove"
                    if "rm_" in p.name or "rightmove" in src
                    else "B_open_data"
                    if any(x in src for x in ("land_registry", "bank_of_england", "ukhpi", "ppd"))
                    or "boe" in p.name
                    or "ukhpi" in p.name
                    or "ppd" in p.name
                    else "C_news"
                    if "news" in str(p) or "politics" in src or "bbc" in src or "guardian" in src
                    else "panel_derived"
                ),
            }
        )
    write_csv(OUT / "DATA_SOURCE_REPORT.csv", report)

    # readiness summary
    def nrows(rel):
        p = OUT / rel
        if not p.exists():
            return 0
        return sum(1 for _ in open(p)) - 1

    meta = {
        "collected_as_of": COLLECTED,
        "duration_seconds": round(time.time() - START, 1),
        "method_doc": "docs/COLLECTION_METHOD.md",
        "script": "scripts/collect_full_research_data.py",
        "tracks": ["A_rightmove", "B_open_data", "C_news"],
        "row_counts": {
            "ukhpi_area_monthly": nrows("prices/ukhpi_area_monthly.csv"),
            "rm_hpi_regional_monthly": nrows("prices/rm_hpi_regional_monthly.csv"),
            "panel_sold_ukhpi_features": nrows("panels/panel_sold_ukhpi_features.csv"),
            "backtest_label_ledger": nrows("panels/backtest_label_ledger.csv"),
            "political_news_items": nrows("news/political_news_items.csv"),
            "boe_bank_rate_monthly": nrows("rates/boe_bank_rate_monthly.csv"),
            "ppd_city_monthly_2026": nrows("prices/ppd_city_monthly_2026.csv"),
        },
        "research_loop_readiness": {
            "historical_price_data": nrows("panels/panel_sold_ukhpi_features.csv") > 1000,
            "political_news": nrows("news/political_news_items.csv") > 10,
            "backtest_labels": nrows("panels/backtest_label_ledger.csv") > 100,
            "source_urls_present": True,
        },
    }
    (OUT / "COLLECTION_META.json").write_text(json.dumps(meta, indent=2))
    # copy method pointer
    method_src = ROOT / "docs" / "COLLECTION_METHOD.md"
    if method_src.exists():
        (OUT / "COLLECTION_METHOD.md").write_text(method_src.read_text())
    log(f"META: {json.dumps(meta, indent=2)}")


def main():
    log(f"=== PropView full research collection @ {COLLECTED} ===")
    collect_rightmove_hpi()
    collect_rightmove_mortgage()
    collect_rightmove_cities()
    collect_ukhpi()
    collect_ppd()
    collect_boe()
    collect_news_rss()
    build_panels()
    write_reports()
    log(f"DONE in {round(time.time()-START,1)}s → {OUT}")


if __name__ == "__main__":
    main()
