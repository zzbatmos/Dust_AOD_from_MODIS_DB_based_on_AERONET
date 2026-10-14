"""Annual branch availability from frozen monthly counts, not mean fractions."""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from paper.numbers import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2017)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Write unpublished arrays outside the public worktree")
    totals, sources, commits = None, {}, set()
    for month in range(1, 13):
        path = args.product_dir / f"DBDust_v2.0.0_Aqua_L3M_1x1_{args.year}{month:02d}.nc"
        sources[str(path.resolve())] = digest(path)
        with xr.open_dataset(path) as ds:
            if ds.sizes["time"] != 1 or int(ds.time.dt.year.item()) != args.year or int(ds.time.dt.month.item()) != month:
                raise ValueError(f"Wrong month in {path}")
            counts = ds[["n_full8", "n_noSSA5", "n_ml", "n_valid_aod"]].isel(time=0, drop=True).astype("int64").load()
            commits.add(ds.attrs["source_code_commit"])
        if totals is None:
            totals = counts
        else:
            totals, counts = xr.align(totals, counts, join="exact")
            totals = totals + counts
    if len(commits) != 1:
        raise ValueError("Monthly products have mixed executable provenance")
    if not bool((totals.n_ml == totals.n_full8 + totals.n_noSSA5).all()):
        raise ValueError("Annual branch counts do not partition ML support")
    if not bool((totals.n_ml <= totals.n_valid_aod).all()):
        raise ValueError("ML support exceeds valid-AOD support")
    totals["full8_fraction_of_valid_aod"] = totals.n_full8 / totals.n_valid_aod.where(totals.n_valid_aod > 0)
    totals["full8_fraction_of_ml"] = totals.n_full8 / totals.n_ml.where(totals.n_ml > 0)
    table = totals.to_dataframe().reset_index()
    table["year"] = args.year
    table["reference"] = "experimental1020"
    table["support"] = "calendar_year_valid_AOD"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"annual_full8_coverage_{args.year}.csv"
    table.to_csv(output, index=False)
    for path, value in sources.items():
        if digest(Path(path)) != value:
            raise RuntimeError(f"Source changed: {path}")
    output.with_suffix(".provenance.json").write_text(json.dumps({
        "output_sha256": digest(output), "script_sha256": digest(Path(__file__)),
        "sources": sources, "product_code_commits": sorted(commits),
        "semantics": "January-December calendar-year sum of full8 counts divided by sum of "
        "valid-AOD counts in each native 1x1 cell; separate fraction on ML support also retained. "
        "Zero denominator stays missing. No averaging of monthly fractions, CALIOP mask, "
        "new screening, or new model inference. Branch availability is not a raw SSA-by-type audit.",
        "global_counts": {v: int(totals[v].sum()) for v in ("n_full8", "n_noSSA5", "n_ml", "n_valid_aod")},
    }, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
