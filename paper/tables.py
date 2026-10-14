"""Export Tables 1 and 2 exclusively from the frozen numbers registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paper.numbers import Registry, digest


def tex(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def number(value, decimals=3):
    return "--" if pd.isna(value) else f"{value:.{decimals}f}"


def table1(registry):
    selections = []
    for population, model, label in (("common", "full8", "full8 (identical rows)"),
                                     ("own", "noSSA5", "noSSA5 (own population)"),
                                     ("own", "hierarchy_prespecified", "Hierarchy (own population)")):
        for tier, threshold in enumerate((.4, .6, .7), 1):
            selections.append((population, model, f"{label}, QA{tier}", threshold))
    for model, label in (("typeconditional", "DB aerosol type"), ("LGEq7_coarseproxy", "LG25"),
                         ("LGscreened_coarseproxy", "LG25 screened"),
                         ("AEfixedthreshold", "AE < 0.6 rule"), ("zero", "Zero")):
        selections.append(("own", model, label, .6))
    metrics = ("unique_rows", "precision", "recall", "FPR", "F1", "AUC", "Brier",
               "DAOD_bias", "DAOD_MAE", "DAOD_RMSE", "DAOD_R2", "EE_descriptive")
    rows, evidence = [], []
    for pop, model, label, threshold in selections:
        filters = dict(reference="experimental1020", population=pop, model=model,
                       scheme="site_cv", QA=threshold, scope="overall", group="all")
        row = {"Method": label, "population": pop}
        for metric in metrics:
            record = registry.number("grouped_validation/metrics", metric, **filters)
            row[metric] = record["value"]
            evidence.append({"row": label, "column": metric, **record})
        source = registry.table("grouped_validation/metrics", **filters).iloc[0]
        row.update(prediction_events=int(source["n"]), sites=int(source["sites"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\caption{Pixel-level skill at held-out sites against the primary AERONET reference.}",
             r"\label{tab:pixel_skill}", r"\begingroup\fontsize{7}{9}\selectfont\setlength{\tabcolsep}{1.5pt}",
             r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
             r"Method & $N$ & Precision & Recall & FPR & F1 & AUC & Brier & Bias & MAE & RMSE & $R^2$ & EE \\", r"\midrule"]
    for row in rows:
        cells = [tex(row["Method"]), str(int(row["unique_rows"]))]
        cells += [number(row[k], 2 if k in ("precision", "recall", "FPR", "F1", "AUC", "EE_descriptive") else 3)
                  for k in metrics[1:]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}\endgroup", r"\end{table}"]
    notes = (
        "Table 1. Primary experimental1020 reference; site-grouped cross-validation. "
        "N counts distinct collocations, not prediction events (two repeats are pooled). "
        "Baseline rows use their own available populations, identified in CSV; screened LG25 "
        "requires SSA and has a smaller population. The DB aerosol-type baseline is the "
        "training-type conditional mean, not a binary DB-dust flag. Baseline classification "
        "metrics use their saved scores/rules at 0.6; deterministic scores are not calibrated "
        "probabilities. DAOD metrics are at 550 nm. EE is the fraction within "
        "+/-(0.05 + 20% of reference DAOD). Intervals and exact source selectors are "
        "provided in Table01_provenance.json. No scores were recomputed from deployment models."
    )
    latex = "\n".join(lines) + "\n"
    latex = latex.replace(r"\end{table}", r"\par\smallskip\begin{minipage}{\linewidth}\fontsize{7}{9}\selectfont "
                          + tex(notes) + r"\end{minipage}" + "\n" + r"\end{table}")
    return df, latex, notes, evidence


def table2(registry):
    names = {"Eselected": "Expected value (this study)", "E1": "QA2 (this study)",
             "E5": "DB aerosol type", "E8": "AE refit (DB space)", "E7": "LG25",
             "Song": "Song et al. (2021) MODIS", "E6": "LG25 screened", "CALIOP": "CALIOP V2"}
    periods = ("DJF", "MAM", "JJA", "SON")
    metrics = ("candidate_mean", "bias", "rmse", "r")
    rows, evidence = [], []
    for support, estimators in (("core_common", ("Eselected", "E1", "E5", "E8", "E7", "Song", "CALIOP")),
                                ("full_common_E6", ("E6", "CALIOP"))):
        for estimator in estimators:
            for period in periods:
                filters = dict(reference="experimental1020", variant="hierarchy", support=support,
                               reference_mask="prior_joint", period=period, region="Land 60S-60N",
                               estimator=estimator)
                row = {"Method": names[estimator], "estimator": estimator, "support": support, "period": period}
                for metric in (*metrics, "reference_mean", "n"):
                    record = registry.number("expected_value_test/stage_c/seasonal_metrics", metric, **filters)
                    row[metric] = record["value"]
                    evidence.append({"row": names[estimator], "period": period, "column": metric, **record})
                rows.append(row)
    df = pd.DataFrame(rows)
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Seasonal land DAOD statistics against CALIOP at 550 nm.}",
             r"\label{tab:global_skill}", r"\begingroup\fontsize{7}{9}\selectfont\setlength{\tabcolsep}{1pt}",
             r"\begin{tabular}{lrrrrrrrrrrrrrrrr}", r"\toprule",
             "Method & " + " & ".join(r"\multicolumn{4}{c}{" + p + "}" for p in periods) + r" \\",
             " & " + " & ".join(["Mean & Bias & RMSE & $r$"] * 4) + r" \\", r"\midrule"]
    for support in ("core_common", "full_common_E6"):
        subset = df[df.support.eq(support)]
        label = "Common support" if support == "core_common" else "Screened-LG25 support (not directly comparable)"
        lines += [r"\multicolumn{17}{l}{" + label + r"} \\"]
        for estimator in subset.estimator.unique():
            q = subset[subset.estimator.eq(estimator)].set_index("period")
            cells = [tex(names[estimator])]
            for period in periods:
                cells += [number(q.loc[period, metric], 2 if metric == "r" else 3) for metric in metrics]
            lines.append(" & ".join(cells) + r" \\")
        q = subset[subset.estimator.eq("CALIOP")].set_index("period")
        lines += ["Cells & " + " & ".join(r"\multicolumn{4}{c}{" + str(int(q.loc[p, "n"])) + "}" for p in periods) + r" \\", r"\midrule"]
    lines += [r"\bottomrule", r"\end{tabular}\endgroup", r"\end{table}"]
    notes = (
        "Table 2. Area-weighted statistics over paired native 2 x 5 degree land cells "
        "between 60S and 60N, using the frozen land-purity, coverage and all-three-month "
        "intersection masks. DJF comprises December 2016-February 2017. Other seasons "
        "are in 2017. CALIOP is spectrally transferred to 550 nm. The screened LG25 block "
        "has different support and its own CALIOP mean/count; comparisons across blocks "
        "are not paired. No statistical confidence intervals are supplied by these tables."
    )
    latex = "\n".join(lines) + "\n"
    latex = latex.replace(r"\end{table}", r"\par\smallskip\begin{minipage}{\linewidth}\fontsize{7}{9}\selectfont "
                          + tex(notes) + r"\end{minipage}" + "\n" + r"\end{table}")
    return df, latex, notes, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Write unpublished tables outside the public worktree")
    registry = Registry.read(args.registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    captions = []
    for index, function in ((1, table1), (2, table2)):
        data, latex, caption, evidence = function(registry)
        prefix = args.output_dir / f"Table{index:02d}"
        data.to_csv(prefix.with_suffix(".csv"), index=False)
        prefix.with_suffix(".tex").write_text(latex)
        prefix.with_name(prefix.name + "_provenance.json").write_text(json.dumps({
            "registry_sha256": digest(args.registry), "script_sha256": digest(Path(__file__)),
            "numbers": evidence}, allow_nan=False, indent=2) + "\n")
        captions.append(caption)
    (args.output_dir / "captions.md").write_text("\n\n".join(captions) + "\n")


if __name__ == "__main__":
    main()
