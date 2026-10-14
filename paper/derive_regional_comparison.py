"""Prepare Fig. 7 from existing seasonal statistics on exactly paired supports."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.numbers import Registry, digest

TABLE = "expected_value_test/stage_c/seasonal_metrics"
METHODS = ("Eselected", "E1", "E5", "E8", "E7", "Song")
SEASONS = ("DJF", "MAM", "JJA", "SON")
REGIONS = ("Sahara", "Middle East", "Central Asia", "East Asian deserts",
           "SW North America", "Sahel", "India", "East China")
SMOKE = (("Amazonia_Cerrado", "SON"), ("southern_Africa", "JJA"),
         ("southern_Africa", "SON"), ("boreal_North_America", "JJA"),
         ("Siberia", "JJA"), ("mainland_SE_Asia", "MAM"),
         ("Amazonia_Cerrado", "DJF"), ("boreal_North_America", "MAM"))
BASE = dict(reference="experimental1020", variant="hierarchy", support="core_common",
            reference_mask="prior_joint")


def derive(registry):
    seasonal = registry.table(TABLE, **BASE)
    rows, smoke = [], []
    for region in REGIONS:
        for estimator in METHODS:
            part = seasonal[seasonal.region.eq(region) & seasonal.estimator.eq(estimator)]
            if len(part) != len(SEASONS) or set(part.period) != set(SEASONS):
                raise ValueError(f"Incomplete seasonal support: {region}/{estimator}")
            part = part.set_index("period").loc[list(SEASONS)]
            ratios = part.candidate_mean / part.reference_mean
            if not np.isfinite(ratios).all() or not (ratios > 0).all():
                raise ValueError("Undefined or nonpositive ratio cannot be silently omitted on log axis")
            row = dict(BASE, region=region, estimator=estimator, seasons=len(SEASONS),
                       ratio_mean=float(ratios.mean()), ratio_min=float(ratios.min()),
                       ratio_max=float(ratios.max()))
            for period in SEASONS:
                row.update({f"{period}_ratio": float(ratios[period]),
                            f"{period}_n": int(part.loc[period, "n"]),
                            f"{period}_candidate_mean": float(part.loc[period, "candidate_mean"]),
                            f"{period}_reference_mean": float(part.loc[period, "reference_mean"])})
            rows.append(row)
    for order, (region, period) in enumerate(SMOKE):
        part = seasonal[seasonal.region.eq(region) & seasonal.period.eq(period)]
        if set(part.estimator) != {*METHODS, "CALIOP"} or len(part) != len(METHODS) + 1:
            raise ValueError(f"Incomplete method comparison: {region}/{period}")
        if part.n.nunique() != 1 or part.reference_mean.nunique() != 1:
            raise ValueError("Methods do not share the same reference/support")
        for estimator in (*METHODS, "CALIOP"):
            row = part[part.estimator.eq(estimator)].iloc[0].to_dict()
            row["row_order"] = order
            smoke.append(row)
    return {"regional_ratio_summary": pd.DataFrame(rows), "regional_smoke_summary": pd.DataFrame(smoke)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Numerical outputs must remain outside the public worktree")
    registry = Registry.read(args.registry)
    table = registry.payload["tables"][TABLE]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in derive(registry).items():
        path = args.output_dir / (name + ".csv")
        if path.exists():
            raise FileExistsError(f"Preserve previous derived outputs: {path}")
        frame.to_csv(path, index=False)
        path.with_suffix(".provenance.json").write_text(json.dumps({
            "output_sha256": digest(path), "script_sha256": digest(Path(__file__)),
            "sources": {table["source_file"]: table["source_sha256"]},
            "input_registry_sha256": digest(args.registry), "base_filter": BASE,
            "semantics": "Ratios: arithmetic mean of four seasonal regional mean ratios to CALIOP; "
            "whiskers are min/max seasonal ratios, NOT confidence intervals. Seasonal n/means/ratios "
            "are retained separately. Absolute comparison rows use the indicated region-season on "
            "core_common/prior_joint support. E6 is excluded because its support differs. "
            "All source metrics are area weighted; no new spatial masks or model predictions.",
        }, indent=2) + "\n")
    print("Wrote verified regional comparison tables")


if __name__ == "__main__":
    main()
