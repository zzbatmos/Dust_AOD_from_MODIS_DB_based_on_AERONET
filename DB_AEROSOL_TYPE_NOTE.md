# MODIS Deep Blue Aerosol-Type Note

## Purpose

This note summarizes how the MODIS Deep Blue (DB) land product appears to treat
`dust` pixels versus other aerosol-type pixels, based on:

- Hsu et al. (2013), doi:`10.1002/jgrd.50712`
- the MODIS Atmosphere QA Plan for Collection 6
- two local Terra MOD04 L2 test granules:
  - heavy dust case: `2025-03-14`
  - heavy smoke case: `2024-09-22`

This is intended as a practical interpretation of the delivered DB product
fields, not a full reverse-engineering of the operational retrieval code.

## Relevant Documents

- [Hsu_etal_2013.pdf](/home/ec2-user/Research/Codex/References/Hsu_etal_2013.pdf)
- [MODIS_QA_Plan.pdf](/home/ec2-user/Research/Codex/References/MODIS_QA_Plan.pdf)

Relevant code used in this repo:

- [plot_modis_db_aerosol_type_flag.py](/home/ec2-user/Research/Codex/plot_modis_db_aerosol_type_flag.py)
- [plot_modis_db_aerosol_type_diagnostics.py](/home/ec2-user/Research/Codex/plot_modis_db_aerosol_type_diagnostics.py)
- [compare_db_flag_vs_two_stage.py](/home/ec2-user/Research/Codex/compare_db_flag_vs_two_stage.py)

## QA Flag Used

The Collection 6 QA plan shows that `Quality_Assurance_Land` contains the
`Deep Blue Aerosol Type` flag. In the local decoding used here:

- bit 0: DB usefulness
- bits 1-2: DB confidence
- bits 3-4: DB aerosol type

The aerosol-type labels are:

- `0 = Mixed`
- `1 = Dust`
- `2 = Smoke`
- `3 = Sulfate`

## Fields Examined

The two-case diagnostic used these standard DB land SDS:

- `Deep_Blue_Spectral_Aerosol_Optical_Depth_Land`
- `Deep_Blue_Spectral_Single_Scattering_Albedo_Land`
- `Deep_Blue_Angstrom_Exponent_Land`

For these Terra MOD04 L2 files:

- AOD is reported at `412`, `470`, and `660 nm`
- SSA is reported at `412`, `470`, and `660 nm`
- AE is reported as one land field

## What the Two Cases Show

### 1. Dust pixels behave like a constrained low-AE branch

For pixels flagged as `Dust`, the delivered AE is nearly fixed near zero.

Heavy dust case `2025-03-14`:

- dust pixels: `723`
- dust AE median: `0.0`
- dust AE range: `0.0` to `0.458`

Heavy smoke case `2024-09-22`:

- dust pixels: `153`
- dust AE median: `0.0`
- dust AE range: `0.0` to `0.423`

This is consistent with the idea that the DB dust branch uses a constrained or
effectively fixed AE for dust-like retrievals.

### 2. Smoke pixels have clearly variable high AE

For pixels flagged as `Smoke`, AE is populated and varies over a broad,
high-value range.

Heavy dust case:

- smoke pixels: `3117`
- smoke AE median: `1.50`
- smoke AE range: `1.21` to `1.80`

Heavy smoke case:

- smoke pixels: `12901`
- smoke AE median: `1.80`
- smoke AE range: `1.205` to `1.80`

This supports the interpretation that non-dust aerosol branches use variable AE
rather than a fixed dust-like value.

### 3. SSA is not exclusive to dust pixels

The delivered DB product does report SSA very often for dust pixels, but SSA is
also present for many smoke and mixed pixels.

Here "data availability fraction" means:

- `number of finite SSA values / number of pixels in that aerosol-type group`

Heavy dust case:

- Dust SSA data availability:
  - `412 nm`: `99.2%`
  - `470 nm`: `72.5%`
  - `660 nm`: `99.9%`
- Smoke SSA data availability:
  - `412 nm`: `18.9%`
  - `470 nm`: `46.2%`
  - `660 nm`: `53.2%`

Heavy smoke case:

- Dust SSA data availability:
  - `412 nm`: `100%`
  - `470 nm`: `99.3%`
  - `660 nm`: `100%`
- Smoke SSA data availability:
  - `412 nm`: `22.2%`
  - `470 nm`: `86.8%`
  - `660 nm`: `87.7%`

So SSA coverage is much stronger for dust pixels, but it is not limited to the
dust branch in the delivered product.

### 4. Spectral AOD is available for all useful aerosol-type categories

In both test granules, the three spectral AOD fields are populated for all
useful aerosol-type pixels:

- Dust
- Smoke
- Mixed

So there is no evidence from these two L2 products that spectral AOD is
restricted to one aerosol type.

## Practical Interpretation

The safest summary is:

- If a pixel is identified as `Dust`, the DB retrieval behaves like a
  constrained dust branch with near-fixed low AE, while reporting spectral AOD
  and usually spectral SSA.
- If a pixel is identified as `Smoke`, `Mixed`, or `Sulfate`, AE is variable
  rather than dust-like.
- However, the output product still often reports SSA for those non-dust pixels,
  so SSA is not exclusive to the dust branch.

## What Not to Say

The following statement is too strong:

- "For dust pixels the algorithm retrieves SSA and AOD, but AE is not reported,
  while for non-dust pixels it retrieves SSA, AE, and AOD."

That is not what the delivered L2 product shows.

A better statement is:

- "For dust pixels, AE is effectively fixed at a low value, while for non-dust
  pixels AE is variable. Spectral AOD is reported for all aerosol types, and
  spectral SSA is most complete for dust pixels but is also often present for
  non-dust pixels."

## Diagnostic Figures

Heavy dust case:

- [aerosol-type map](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_db_aerosol_type_map.png)
- [AOD histograms](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_db_aerosol_type_aod_histograms.png)
- [SSA histograms](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_db_aerosol_type_ssa_histograms.png)
- [AE histogram](/home/ec2-user/Research/Codex/dust_model_application_2025-03-14/MOD04_L2.A2025073.1705.061.2025074013638_db_aerosol_type_ae_histogram.png)

Heavy smoke case:

- [aerosol-type map](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127_db_aerosol_type_map.png)
- [AOD histograms](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127_db_aerosol_type_aod_histograms.png)
- [SSA histograms](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127_db_aerosol_type_ssa_histograms.png)
- [AE histogram](/home/ec2-user/Research/Codex/dust_model_application_2024-09-22_amazon/MOD04_L2.A2024266.1345.061.2024268021127_db_aerosol_type_ae_histogram.png)
