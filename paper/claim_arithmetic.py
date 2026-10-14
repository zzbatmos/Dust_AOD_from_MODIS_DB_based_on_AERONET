"""Descriptive claim arithmetic from a private frozen Registry, without fitting.

Run with python -m paper.claim_arithmetic --registry INPUT --output-file OUTPUT.
OUTPUT must be outside the public package worktree. Records retain unrounded
values and complete Registry.number operands; they are not new experiments.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from paper.numbers import Registry, digest


PRIMARY = "experimental1020"
SEASONS = ("DJF", "MAM", "JJA", "SON")
DESERTS = ("Sahara", "Middle East")
MIXED_REGIONS = ("Sahel", "India", "East China")
SMOKE_SEASONS = (
    ("Amazonia_Cerrado", "SON"),
    ("boreal_North_America", "JJA"),
    ("Siberia", "JJA"),
    ("southern_Africa", "JJA"),
    ("southern_Africa", "SON"),
)
DECOMPOSITION = "grouped_validation/decomposition_paired_differences"
SITE_MONTH = "expected_value_test/stage_b/site_month_metrics"
SEASONAL = "expected_value_test/stage_c/seasonal_metrics"
LOW_LOADING = "expected_value_test/stage_c/low_loading_contributions"
SITE_FILTER = dict(
    stage="b", reference=PRIMARY, branch="hierarchy_own", scheme="site_cv",
    pairing="core", stratum="all", repeat_summary="mean_over_repeats", minimum=3,
)
SEASON_FILTER = dict(
    reference=PRIMARY, variant="hierarchy", support="core_common",
    reference_mask="prior_joint",
)
SEASON_CAVEATS = (
    "Native 2x5 seasonal cells on prior_joint core_common support; land purity "
    ">=0.95 and fine-land coverage >=0.5; equal three-month means, then land-area weighting.",
    "DJF includes December 2016; the four seasons span December 2016-November 2017.",
    "Descriptive comparison to CALIOP, not truth or a calibrated uncertainty interval.",
)


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite numeric value, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite numeric value, got {value!r}")
    return float(value)


def _operand(registry, table, metric, **filters):
    number = registry.number(table, metric, **filters)
    _finite(number["value"], f"{table}/{metric}")
    if not number["source_sha256"] or not number["N"]:
        raise ValueError(f"{table}/{metric}: missing source hash or counts")
    return {**number, "table": table, "metric": metric, "requested_filter": filters}


def _same_counts(operands):
    counts = [operand["N"] for operand in operands.values()]
    if any(count != counts[0] for count in counts[1:]):
        raise ValueError("Operand counts differ; cannot claim paired support")


def _positive(value, label):
    value = _finite(value, label)
    if value <= 0:
        raise ValueError(f"{label}: denominator must be positive")
    return value


def _record(key, value, operands, equation, caveats, *, units="1", interval=None,
            interval_type=None):
    if not isinstance(value, (str, bool)):
        _finite(value, key)
    _same_counts(operands)
    return {
        "key": key, "value": value, "units": units, "interval": interval,
        "interval_type": interval_type,
        "N": {role: operand["N"] for role, operand in operands.items()},
        "operands": operands, "equation": equation, "caveats": list(caveats),
    }


def _season_operand(registry, region, period, estimator, metric, reference=PRIMARY):
    return _operand(
        registry, SEASONAL, metric,
        **{**SEASON_FILTER, "reference": reference, "region": region,
           "period": period, "estimator": estimator},
    )


def _season_comparison(registry, key, region, period, estimator, extra=()):
    operands = {
        "candidate_mean": _season_operand(
            registry, region, period, estimator, "candidate_mean"),
        "reference_mean": _season_operand(
            registry, region, period, estimator, "reference_mean"),
    }
    candidate = operands["candidate_mean"]["value"]
    reference = _positive(operands["reference_mean"]["value"], key)
    ratio = candidate / reference
    caveats = (*SEASON_CAVEATS, *extra)
    return [
        _record(key + "/ratio", ratio, operands,
                "candidate_mean / reference_mean", caveats),
        _record(key + "/relative_bias_percent", 100 * (ratio - 1), operands,
                "100 * (candidate_mean / reference_mean - 1)", caveats, units="%"),
        _record(key + "/bias", candidate - reference, operands,
                "candidate_mean - reference_mean", caveats, units="DAOD550"),
    ]


def derive_claims(registry: Registry) -> list[dict]:
    """Resolve every required operand exactly; missing/ambiguous inputs fail closed.

    N is indexed by operand role rather than merging unlike sample definitions.
    Marginal input intervals are preserved as provenance, never combined into a
    ratio interval without the original joint bootstrap draws.
    """
    records = []
    for target, a, b in (
        ("size", "DB_size500proxy", "AER_size500proxy"),
        ("depolarization", "DB_nonsphericity", "AER_nonsphericity"),
    ):
        operand = _operand(
            registry, DECOMPOSITION, "delta_RMSE_A_minus_B",
            reference=PRIMARY, population="decomposition", scheme="site_cv", QA=0.6,
            A=a, B=b,
        )
        interval = operand["interval"]
        if interval is None or len(interval) != 2:
            raise ValueError("Decomposition requires its saved paired interval")
        lo, hi = (_finite(x, "decomposition interval") for x in interval)
        if lo > hi:
            raise ValueError("Decomposition interval endpoints are reversed")
        records.append(_record(
            f"decomposition/{target}/rmse_reduction", -operand["value"],
            {"delta_RMSE_A_minus_B": operand},
            "-delta_RMSE_A_minus_B; interval = [-source_hi, -source_lo]",
            ("Sign reversal of a saved paired site-bootstrap contrast; no refitting.",
             "Two-predictor controlled comparison, repeats pooled; size target is SDA CMF500proxy."),
            units="DAOD550", interval=[-hi, -lo], interval_type=operand["interval_type"],
        ))

    ev = _operand(registry, SITE_MONTH, "rmse", **SITE_FILTER, estimator="Eselected")
    for estimator in ("E1", "E5"):
        baseline = _operand(registry, SITE_MONTH, "rmse", **SITE_FILTER, estimator=estimator)
        denominator = _positive(baseline["value"], "baseline_rmse")
        if ev["value"] < 0:
            raise ValueError("EV RMSE must be nonnegative")
        records.append(_record(
            f"site_month/Eselected_vs_{estimator}/rmse_improvement_percent",
            100 * (1 - ev["value"] / denominator),
            {"ev_rmse": ev, "baseline_rmse": baseline},
            "100 * (1 - ev_rmse / baseline_rmse)",
            ("Primary site-CV hierarchy, core pairing, minimum three distinct collocations per cell.",
             "Ratio of mean-over-repeats RMSEs, not mean of repeat-specific improvement ratios.",
             "Eselected is fold-specific E3c/E4 selection, not solely a two-part mixture.",
             "No ratio CI inferred from marginal RMSE intervals; small bias is not equivalence to zero."),
            units="%",
        ))

    for period in SEASONS:
        records.extend(_season_comparison(
            registry, f"global/{period}/Eselected", "Land 60S-60N", period, "Eselected"))

    for region in DESERTS:
        for period in ("MAM", "JJA", "SON"):
            for estimator in ("E1", "Eselected", "E8"):
                records.extend(_season_comparison(
                    registry, f"desert/{region}/{period}/{estimator}",
                    region, period, estimator))
            qa2 = _season_operand(registry, region, period, "E1", "candidate_mean")
            ev_mean = _season_operand(registry, region, period, "Eselected", "candidate_mean")
            reference = _season_operand(registry, region, period, "E1", "reference_mean")
            ev_reference = _season_operand(
                registry, region, period, "Eselected", "reference_mean")
            if reference["value"] != ev_reference["value"]:
                raise ValueError("Desert estimators do not share the same reference mean")
            operands = dict(qa2_mean=qa2, ev_mean=ev_mean, reference_mean=reference,
                            ev_reference_mean=ev_reference)
            difference = abs(qa2["value"] - reference["value"]) - abs(
                ev_mean["value"] - reference["value"])
            caveats = (*SEASON_CAVEATS,
                       "Positive difference means EV is closer in absolute regional-mean bias, not RMSE.",
                       "Sahara MAM is an audited exception candidate; the flag is calculated, not hardcoded.")
            key = f"desert/{region}/{period}"
            records.append(_record(
                key + "/qa2_minus_ev_absolute_bias", difference, operands,
                "abs(qa2_mean - reference_mean) - abs(ev_mean - reference_mean)",
                caveats, units="DAOD550"))
            records.append(_record(
                key + "/qa2_as_close_or_closer_exception", difference > 0, operands,
                "abs(qa2_mean - reference_mean) > abs(ev_mean - reference_mean)",
                caveats, units="boolean"))

    for region, period in SMOKE_SEASONS:
        for estimator in ("Eselected", "E7", "Song"):
            records.extend(_season_comparison(
                registry, f"smoke/{region}/{period}/{estimator}", region, period, estimator,
                ("Primary boreal boxes end at 60N; not the 50-75N sensitivity boxes.",
                 "Absolute DAOD bias is not percent error; a 2-6-fold statement does not apply universally, including southern Africa.")))
        for estimator in ("E7", "Song"):
            published = _season_operand(registry, region, period, estimator, "candidate_mean")
            ev_mean = _season_operand(registry, region, period, "Eselected", "candidate_mean")
            operands = dict(published_mean=published, ev_mean=ev_mean)
            ratio = published["value"] / _positive(ev_mean["value"], "smoke_ev_mean")
            caveats = (*SEASON_CAVEATS,
                       "Published partition compared to EV, not to CALIOP; this is not a truth-error metric.")
            key = f"smoke/{region}/{period}/{estimator}_vs_EV"
            records.append(_record(key + "/ratio", ratio, operands,
                                   "published_mean / ev_mean", caveats))
            records.append(_record(
                key + "/mean_difference", published["value"] - ev_mean["value"],
                operands, "published_mean - ev_mean", caveats, units="DAOD550"))

    for region in MIXED_REGIONS:
        for period in SEASONS:
            for estimator in ("Eselected", "E7", "Song"):
                key = f"mixed/{region}/{period}/{estimator}"
                comparisons = _season_comparison(registry, key, region, period, estimator)
                records.extend(comparisons)
                ratio_record = comparisons[0]
                records.append(_record(
                    key + "/underestimation_exception", ratio_record["value"] >= 1,
                    ratio_record["operands"], "candidate_mean >= reference_mean",
                    (*SEASON_CAVEATS,
                     "Flags equality or overestimation, contradicting strict underestimation; no dust-definition or sampling attribution follows."),
                    units="boolean"))

    for period in SEASONS:
        old = _season_operand(registry, "Land 60S-60N", period, "Eselected",
                              "candidate_mean", reference="old675LR56")
        primary = _season_operand(registry, "Land 60S-60N", period, "Eselected",
                                  "candidate_mean")
        ratio = old["value"] / _positive(primary["value"], "primary_mean")
        operands = dict(old_mean=old, primary_mean=primary)
        caveats = (*SEASON_CAVEATS,
                   "Reference-scenario sensitivity changes wavelength, lidar ratio, transfer, and target-dependent models.",
                   "Not an isolated lidar-ratio effect, calibrated uncertainty, or confidence interval; matching counts alone do not prove identical cell identities.")
        records.append(_record(
            f"reference_sensitivity/{period}/old_to_primary_ratio", ratio, operands,
            "old_mean / primary_mean", caveats))
        records.append(_record(
            f"reference_sensitivity/{period}/increase_percent", 100 * (ratio - 1),
            operands, "100 * (old_mean / primary_mean - 1)", caveats, units="%"))

    filters = dict(reference=PRIMARY, variant="hierarchy", support="core_common",
                   estimator="Eselected", region="Land 60S-60N", subset="AOD550_lt_0p2")
    low = _operand(registry, LOW_LOADING, "area_weighted_mean", **filters)
    full = _operand(registry, LOW_LOADING, "full_mean_DAOD", **filters)
    fraction = low["value"] / _positive(full["value"], "full_mean_DAOD")
    operands = dict(low_aod_contribution=low, full_mean_DAOD=full)
    caveats = (
        "Calendar 2017 MODIS-only covered cell-months, area weighted; not seasonal prior_joint support.",
        "Not a global retrieval-sum fraction, scene frequency, or accuracy validation of low-AOD extrapolation.",
    )
    records.append(_record(
        "low_aod/contribution_fraction", fraction, operands,
        "low_aod_contribution / full_mean_DAOD", caveats))
    records.append(_record(
        "low_aod/contribution_percent", 100 * fraction, operands,
        "100 * low_aod_contribution / full_mean_DAOD", caveats, units="%"))
    if len({record["key"] for record in records}) != len(records):
        raise ValueError("Duplicate derived claim keys")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.registry.resolve()
    output = args.output_file.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Numerical outputs must be outside the public code worktree")
    if output == source:
        raise ValueError("Output must not overwrite the input registry")
    source_hash = digest(source)
    records = derive_claims(Registry.read(source))
    if digest(source) != source_hash:
        raise ValueError("Input registry changed during arithmetic")
    payload = {
        "schema_version": 1, "registry_file": str(source), "registry_sha256": source_hash,
        "generator_sha256": digest(Path(__file__)),
        "scope": "Arithmetic on frozen Registry.number operands only; no fitting or new experiments",
        "records": records,
    }
    serialized = json.dumps(payload, allow_nan=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)
    print(json.dumps({"records": len(records), "output_file": str(output)}))


if __name__ == "__main__":
    main()
