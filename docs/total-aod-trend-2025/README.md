# MODIS Deep Blue Total AOD Trend Web Report

This folder contains a GitHub Pages-ready static report for the MODIS Deep Blue total AOD trend and recent anomaly analysis.

Open locally:

```bash
python -m http.server 8000 --directory docs
```

Then visit:

```text
http://localhost:8000/total-aod-trend-2025/
```

If GitHub Pages is enabled from the repository `docs/` folder, the report should be available at:

```text
https://zzbatmos.github.io/Dust_AOD_from_MODIS_DB_based_on_AERONET/total-aod-trend-2025/
```

Only selected figures and compact CSV summaries are included here. The full Level-3 production data are not included in this web report folder.

## June 28, 2026 Update

The report now includes an added section for the 2025 single-year anomaly and an operational `MOD08_M3`/`MYD08_M3` quick-look check. This update keeps the earlier 2024-2025 two-year anomaly analysis and adds:

- 2025-only production-freeze anomaly/significance maps.
- 2025 monthly and annual anomaly strength figure.
- 2025 regional contribution decomposition.
- Operational monthly MODIS M3 Dark Target, Deep Blue, and combined AOD corroboration.
- Production-freeze versus operational M3 Deep Blue annual comparison.
