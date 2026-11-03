# The Killing Cascade: Replication Materials

Code and data for:

> Nix, J., & Adams, I. T. (in press). The killing cascade: More officers, more wounds, more deaths in police shootings. *Police Quarterly*. https://doi.org/10.1177/10986111261507732

**Preprint:** [CrimRxiv](https://www.crimrxiv.com/pub/7mj8aj3g)

The paper studies 4,207 civilians shot by police in Texas (September 2015 to December 2025) and California (January 2016 to December 2025). It asks how case fatality varies with the number of officers and, in California, with where civilians were wounded. The analysis is descriptive. See the paper for what the estimates do and do not show.

## Quick start

```bash
pip install -r requirements.txt
python analysis/scripts/run_all.py
```

The run takes about 10 minutes, most of it the 5,000-draw agency-cluster bootstrap. Every table, figure, and inline statistic is written to `analysis/outputs/`. The committed copy of `analysis/outputs/` is the run behind the accepted manuscript, so you can compare your results against it.

The county maps (Figures 5 and 6) use `geopandas` and `adjustText`, and they download the Census TIGER county shapefile. Skip them with `--skip-maps`.

## Repository layout

```
data/processed/         analysis-ready inputs (see "Data" below)
analysis/scripts/
  study1_texas/         Texas (Study 1)
  study2_california/    California (Study 2)
  revision/             supplementary analyses and data-validation checks
  pooled_analysis.py    pooled two-state models
  create_figures.py     Figures 1-4
  create_map_panels.py  Figures 5-6
  lpm_utils.py          shared estimator (LPM, year/agency FE, agency-clustered SEs)
  run_all.py            runs everything in order
analysis/outputs/       results (CSV) and figures (PNG/PDF)
docs/source_data_notes.md   quirks in the raw CA DOJ files
```

## Data

All inputs are public records. This repository includes **processed** files only. The `00_*` scripts that built them from the raw downloads are included for transparency, but they are not run by `run_all.py`.

| File | Source | Built by |
|---|---|---|
| `texas_ois_2015_2025.csv` | Texas Office of the Attorney General officer-involved shooting reports. These come from the [Texas Justice Initiative](https://texasjusticeinitiative.org/) compilation through Aug. 26, 2024, and were parsed from the [OAG portal](https://oag.my.site.com/oisreports) after that date. | `study1_texas/00_build_oag_extension.py` → `00b_validate_and_collapse_oag.py` → `00c_build_texas_file.py` |
| `tx_oag_injury_of_officer.csv` | OAG reports of officers shot by civilians (used to flag civilians who shot an officer) | `00_build_oag_extension.py` |
| `tx_oag_reports_parsed.csv`, `tx_oag_persons_all.csv` | Parsed OAG reports, and the reports grouped into persons; used by the Texas linkage check (`revision/18`) | `00_build_oag_extension.py`, `00b_validate_and_collapse_oag.py` |
| `master_*_clean.csv`, `master_ori_agency_mapping.csv` | California DOJ [URSUS use-of-force data](https://openjustice.doj.ca.gov/data), 2016-2025 | `study2_california/00_build_master_files.py` |

`analysis/outputs/study1/tables/tx_data_construction.csv` is also a product of the build step: counts of records linked, merged, and added while assembling the Texas file. It is committed because the paper reports it.

**Redactions.** In the Texas files, the following fields are blank: civilian names, incident street addresses, narrative text, and the names and email addresses of the agency staff who filed each report. No analysis uses them, and they remain available from the original sources. Every other field is as built.

### Key measures

- **Officer count differs by state.** Texas counts the officers listed on a person's OAG reports, which we read as the officers who fired (`Officer_Count`). California counts officers *involved* (URSUS `NUM_INVOLVED_OFFICERS`), which can include officers who did not fire.
- **Wound location (California only).** Each civilian is assigned the most severe region struck, from an exact lookup of the 11 URSUS body regions. "Regions wounded" counts distinct regions; it is not a count of gunshot wounds.
- **Primary specification.** Linear probability models with incident-year fixed effects and standard errors clustered by agency; Model 5 adds agency fixed effects.

## Where each result comes from

| Paper | Output file(s) in `analysis/outputs/` | Script |
|---|---|---|
| Table 1 | `study1/texas_analysis_ready.csv`, `study2/california_analysis_ready.csv`, `*/tables/cfr_by_officer_count.csv` | `study*/01_prepare_data.py`, `study*/02_run_analysis.py` |
| Table 2 (predicted fatality by officer count) | `revision/predicted_fatality_by_officers.csv` | `revision/10_predicted_fatality.py` |
| Table 3 (Texas models) | `study1/tables/regression_models.csv` | `study1_texas/02_run_analysis.py` |
| Table 4 (California models) | `study2/tables/regression_models.csv` | `study2_california/02_run_analysis.py` |
| Table 5 (rear-only wounds) | `revision/rear_wound_race_models.csv`, `rear_wound_descriptives.csv` | `revision/13_rear_wound_race.py` |
| Figure 1 | `figures/fig1_cfr_by_officer_count.png` | `create_figures.py` |
| Figure 2 | `figures/fig6_lethal_cascade.png` | `create_figures.py` |
| Figure 3 | `figures/fig2_forest_plot.png` | `create_figures.py` |
| Figure 4 | `figures/fig3_robustness_curve.png` | `create_figures.py` (from `*/tables/robustness_results.csv`) |
| Figures 5-6 | `figures/fig_texas_map_panels.png`, `fig_california_map_panels.png` | `create_map_panels.py` |
| Tables A1-A3 (logits) | `*/tables/regression_models_logistic.csv` | `study*/02_run_analysis.py`, `pooled_analysis.py` |
| Table A4 (expected-death differences) | `revision/expected_death_differences.csv` | `revision/10_predicted_fatality.py` |
| Table A5 (race × wound location) | `revision/race_wound_crosstab.csv`, `race_wound_chisq_variants.csv` | `revision/05_race_wound_crosstab.py` |
| Table A6 (decomposition) | `study2/tables/mediation_results.csv`, `revision/bootstrap_mediation_results.csv` | `study2_california/03_mediation_analysis.py`, `revision/02_bootstrap_mediation.py` |
| Table A7 (functional form) | `revision/nonlinear_officer_models_*.csv` | `revision/01_nonlinear_officer_count.py` |
| Table A8 (sample flow, missing data) | `revision/sample_flow.csv`, `excluded_vs_included.csv`, `ipw_missing_sensitivity.csv` | `revision/12_sample_flow.py` |
| Table A9 (pooled) | `pooled/tables/regression_models.csv`, `revision/state_*_interaction.csv` | `pooled_analysis.py`, `revision/03_pooled_interactions.py` |
| Table A10 (balance, Oster) | `revision/placebo_test_results.csv`, `oster_bounds.csv` | `revision/07_placebo_falsification.py`, `revision/09_oster_bounds.py` |
| Table A11 (moderators) | `revision/mechanism_tests.csv` | `revision/08_mechanism_discrimination.py` |
| Robustness specifications | `*/tables/robustness_results.csv` | `study1_texas/03_robustness.py`, `study2_california/04_robustness.py` |
| Texas race heterogeneity (M4/M5) | `revision/race_heterogeneity_fe.csv` | `revision/19_race_heterogeneity_fe.py` |

The other `revision/` scripts produce checks reported in footnotes or in our responses to reviewers:
- collinearity (`14`);
- single-case agencies (`15`);
- the California agency-ID audit (`17`);
- Texas linkage sensitivity (`18`).

If you work from the raw CA DOJ downloads, read [`docs/source_data_notes.md`](docs/source_data_notes.md) first. It covers problems in the public files that you'll need to handle, including a 2022 armed-status flag that contradicts the recorded weapon.

## Environment

The results were produced with Python 3.13 (pandas 3.0, statsmodels 0.15) on Windows 11. `requirements.txt` pins the exact versions.

## License

Code: MIT (see `LICENSE`). Processed data: CC BY 4.0, subject to the terms of the original public sources. Please cite the paper if you use these materials (see `CITATION.cff`).

## Contact

Justin Nix, University of Nebraska Omaha (jnix@unomaha.edu)
Ian T. Adams, University of South Carolina
