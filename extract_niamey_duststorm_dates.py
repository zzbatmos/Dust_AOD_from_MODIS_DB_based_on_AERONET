#!/usr/bin/env python3
"""Extract Niamey duststorm dates from timeanddate.com monthly history pages."""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests


URL_TEMPLATE = "https://www.timeanddate.com/weather/niger/niamey/historic?month={month}&year={year}"
DATE_RE = re.compile(r'"ds":"([^"]+)","icon":\d+,"desc":"Duststorm\."', re.DOTALL)
MONTH_RE = re.compile(r"<title>Weather in ([A-Z][a-z]+ \d{4}) in Niamey, Niger</title>")
AVAILABLE_MONTHS_RE = re.compile(r'Select month:.*?<select[^>]*id=month[^>]*>(.*?)</select>', re.DOTALL)
MONTH_VALUE_RE = re.compile(r'value=(\d{4})-(\d{2})')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2002)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ec2-user/Research/Codex/niamey_duststorm_dates"),
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    return parser.parse_args()


def fetch_month(session: requests.Session, year: int, month: int) -> str:
    url = URL_TEMPLATE.format(month=month, year=year)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def page_has_requested_month(html: str, year: int, month: int) -> bool:
    match = MONTH_RE.search(html)
    if not match:
        return False
    shown = datetime.strptime(match.group(1), "%B %Y")
    return shown.year == year and shown.month == month


def extract_dates(html: str) -> list[str]:
    dates = sorted(
        {
            datetime.strptime(match.split(", ", 1)[1].rsplit(", ", 1)[0], "%B %d, %Y").strftime("%Y-%m-%d")
            for match in DATE_RE.findall(html)
        }
    )
    return dates


def available_months(session: requests.Session) -> set[tuple[int, int]]:
    html = fetch_month(session, 4, 2017)
    block_match = AVAILABLE_MONTHS_RE.search(html)
    if not block_match:
        raise RuntimeError("Could not parse available month list from reference page.")
    months: set[tuple[int, int]] = set()
    for year, month in MONTH_VALUE_RE.findall(block_match.group(1)):
        months.add((int(year), int(month)))
    return months


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; Codex research workflow; +https://github.com/zzbatmos/Dust_AOD_from_MODIS_DB_based_on_AERONET)"
        }
    )

    rows: list[dict[str, object]] = []
    missing_months: list[str] = []
    all_dates: list[str] = []
    available = available_months(session)

    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            if (year, month) not in available:
                missing_months.append(f"{year:04d}-{month:02d}")
                continue
            html = fetch_month(session, year, month)
            if not page_has_requested_month(html, year, month):
                missing_months.append(f"{year:04d}-{month:02d}")
                time.sleep(args.sleep_seconds)
                continue
            dates = extract_dates(html)
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "n_duststorm_dates": len(dates),
                    "duststorm_dates": ";".join(dates),
                }
            )
            all_dates.extend(dates)
            time.sleep(args.sleep_seconds)

    csv_path = args.output_dir / f"niamey_duststorm_dates_{args.start_year}_{args.end_year}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "month", "n_duststorm_dates", "duststorm_dates"])
        writer.writeheader()
        writer.writerows(rows)

    date_list_path = args.output_dir / f"niamey_duststorm_unique_dates_{args.start_year}_{args.end_year}.txt"
    unique_dates = sorted(set(all_dates))
    date_list_path.write_text("\n".join(unique_dates) + ("\n" if unique_dates else ""), encoding="ascii")

    year_counts = Counter(date[:4] for date in unique_dates)
    summary_path = args.output_dir / f"niamey_duststorm_summary_{args.start_year}_{args.end_year}.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Niamey Duststorm Dates ({args.start_year}-{args.end_year})\n\n")
        f.write(f"- Source: {URL_TEMPLATE.format(month=4, year=2017)}\n")
        f.write(f"- Unique dates with at least one `Duststorm.` record: `{len(unique_dates)}`\n")
        f.write(f"- Months successfully parsed: `{len(rows)}`\n")
        f.write(f"- Months not available / did not match requested month: `{len(missing_months)}`\n\n")
        if missing_months:
            f.write("## Missing Months\n\n")
            f.write(", ".join(missing_months) + "\n\n")
        f.write("## Yearly Counts\n\n")
        for year in range(args.start_year, args.end_year + 1):
            f.write(f"- `{year}`: `{year_counts.get(str(year), 0)}` duststorm dates\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {date_list_path}")
    print(f"Wrote {summary_path}")
    print(f"Unique dates: {len(unique_dates)}")
    print(f"Missing months: {len(missing_months)}")


if __name__ == "__main__":
    main()
