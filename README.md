[README.md](https://github.com/user-attachments/files/35515773/README.md)
# Parkinson’s Disease Progression Prediction

CS4486 AI Mini Project · Group 2 · Topic 5

This project reproduces and improves a baseline for the Kaggle AMP®-Parkinson’s Disease Progression Prediction competition. It predicts four MDS-UPDRS scores (`updrs_1` to `updrs_4`) at the current visit and 6, 12, and 24 months ahead. The competition uses SMAPE+1, a symmetric percentage error computed after adding 1 to both the observed and predicted scores. Lower scores are better.

## Repository

[`Code.py`](Code.py) contains the final **V4** submission pipeline, from training and feature construction to streaming prediction. V1–V3 appear in the results table for comparison.

## Final model

- **Features.** Build visit-schedule and sampling indicators, then summarize longitudinal protein and peptide measurements. During inference, update the history as Kaggle releases batches.
- **Tabular ensemble.** Average the probabilities of three multiclass LightGBM models and decode the integer score that minimizes expected SMAPE+1. Blend those predictions with three PyTorch MLPs.
- **Multi-view neural networks.** Train two models with different seeds for each of four output layouts. The layouts predict one score, four scores at one horizon, four horizons for one score, or all 16 score–horizon combinations.
- **Final fusion.** Combine the tabular ensemble, multi-view predictions, and cohort-based biomarker trajectories using target-specific weights. Round and clip final scores to the range 0–86.

All models are trained from scratch.

## Results

These are the team’s reported Kaggle submission scores.

| Version | Main change | Public SMAPE+1 ↓ | Private SMAPE+1 ↓ |
| --- | --- | ---: | ---: |
| V1 | Histogram gradient boosting with a target-month trend lookup | 58.188 | 70.326 |
| V2 | Visit-schedule features, LightGBM minimum-risk decoding, and a target-routed MLP | 56.812 | 61.306 |
| V3 | Compact features, a deeper LeakyReLU MLP, and tree–MLP averaging | 54.662 | **60.058** |
| V4 | Longitudinal biomarkers, multi-view networks, and cohort-aware fusion | **53.748** | 60.099 |

V4 improved the baseline by **4.440** points on the public split and **10.227** points on the private split. V3 had the lowest private score, beating V4 by **0.041** points.

## Run V4 on Kaggle

1. Create a notebook for the competition and add the official [competition data](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data). The script expects the data at `/kaggle/input/amp-parkinsons-disease-progression-prediction`.
2. Use the CPU runtime with `numpy`, `pandas`, `lightgbm`, and `torch` available. The team ran the project on Python 3.10.10.
3. Paste the full contents of [`Code.py`](Code.py) into a notebook code cell and run it. The script trains the ensemble and passes each prediction batch to the competition API through `environment.predict()`.

The script reads `train_clinical_data.csv`, `supplemental_clinical_data.csv`, `train_proteins.csv`, and `train_peptides.csv` from the mounted data directory. It first tries to load `amp_pd_peptide_310` and falls back to `amp_pd_peptide`. The submission entry point requires the Kaggle competition environment.

## Team

- **LI CECILIA** — Group leader
- **SONG TINGYI** — Group member

## References

- Competition task, dataset, and evaluation. [Kaggle competition page](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction)
- Visit-schedule features and metric-aligned decoding. [Connecting dotts… 1st Place Solution](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/writeups/connecting-dotts-1st-place-solution)
- Visit frequency as a signal for identifying the control cohort. [Ambrosm, Find the Control Group](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/writeups/ambrosm-5-find-the-control-group)
