from __future__ import annotations

import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

HORIZONS = (0, 6, 12, 24)
VISIT_MONTHS = (0, 6, 12, 18, 24, 36, 48, 60, 72, 84)
CSF_TRACK_MONTHS = (0, 6, 12, 18, 24)
TARGETS = tuple(f"updrs_{i}" for i in range(1, 5))

PREDICTION_ID_PATTERN = re.compile(
    r"^(?P<patient>\d+)_(?P<visit>-?\d+)_updrs_(?P<target>[1-4])_plus_(?P<horizon>\d+)_months$"
)

TREE_SEEDS = (42, 1024, 2048)
TABULAR_NN_SEEDS = (1001, 1002, 1003)
MULTIVIEW_KINDS = ("scalar", "target4", "horizon4", "joint16")
MULTIVIEW_SEEDS = (3101, 3102)

# Target-specific weights for multi-view neural network vs tabular GBDT/MLP
TARGET_MULTIVIEW_WEIGHT = {1: 0.60, 2: 0.00, 3: 0.80, 4: 0.20}
# Target-specific weights for fusing the Cohort-Biomarker trajectory with the ML ensemble
TARGET_BIO_TRAJECTORY_WEIGHT = {1: 0.55, 2: 0.65, 3: 0.60, 4: 0.75}


def decode_minimum_risk(probabilities: np.ndarray) -> np.ndarray:
    """Bayes minimum-risk decoder under the competition SMAPE+1 cost matrix."""
    scores = np.arange(probabilities.shape[1], dtype=np.float64)
    costs = 2.0 * np.abs(scores[:, None] - scores[None, :])
    costs /= 2.0 + scores[:, None] + scores[None, :]
    return np.argmin(probabilities @ costs, axis=1)


def build_progression_curves() -> dict[tuple[int, int, str], int]:
    """
    Constructs cohort-stratified longitudinal base progression curves:
      - cohort_group in {0, 1, 2}: PD clinical progression schedules across 0..144 months
      - cohort_group == 3: Strictly annual healthy control schedule (UPDRS 2..4 = 0)
    """
    init_scores = {
        "updrs_1": 4,
        "updrs_2": 3,
        "updrs_3": 17,
        "updrs_4": 0,
    }
    step_months = {
        "updrs_1": {3, 6, 18, 72, 84, 90, 96, 102, 108},
        "updrs_2": {1, 3, 6, 30, 42, 72, 84, 90, 96, 102, 108},
        "updrs_3": {1, 3, 6, 12, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108},
        "updrs_4": {72, 78, 84, 96, 108},
    }
    curves: dict[tuple[int, int, str], int] = {}

    for month in range(145):
        for cohort in (0, 1, 2):
            for target in TARGETS:
                if month == 0:
                    curves[(month, cohort, target)] = init_scores[target]
                else:
                    prev = curves[(month - 1, cohort, target)]
                    inc = 1 if month in step_months[target] else 0
                    curves[(month, cohort, target)] = prev + inc

        # Cohort 3: Annual-only visitor cohort (enriched for healthy controls)
        curves[(month, 3, "updrs_1")] = 3 if month >= 48 else 2
        curves[(month, 3, "updrs_2")] = 0
        curves[(month, 3, "updrs_3")] = 0
        curves[(month, 3, "updrs_4")] = 0

    return curves


PROGRESSION_CURVES = build_progression_curves()


def causal_percentile_rank(
    df: pd.DataFrame, value_col: str, rank_col: str
) -> pd.DataFrame:
    """Computes causal cross-sectional percentile rank using only visits <= current month."""
    months = sorted(df["visit_month"].unique())
    blocks = []
    for m in months:
        cur = df[df["visit_month"] == m].copy()
        pool = df[df["visit_month"] <= m]
        ranks = pool[value_col].rank(method="average", ascending=False, pct=True)
        cur[rank_col] = ranks.loc[cur.index].values
        blocks.append(cur)
    return pd.concat(blocks, axis=0).sort_index()


class BiomarkerTrajectoryEngine:
    """
    Extracts scale-invariant (SMAPE-normalized), causally rank-normalized protein/peptide
    composite indices and patient visit-interval cohort groups.
    Pre-pivots static training matrices once for fast online inference.
    """

    def __init__(
        self,
        train_clinical: pd.DataFrame,
        train_proteins: pd.DataFrame,
        train_peptides: pd.DataFrame,
    ) -> None:
        self.base_clinical = (
            train_clinical[["patient_id", "visit_month", "visit_id"]]
            .drop_duplicates()
            .copy()
        )
        self.pro_counts_base = (
            train_proteins.groupby(["patient_id", "visit_month"])["NPX"]
            .count()
            .rename("NPX_count")
            .reset_index()
        )
        self.pep_counts_base = (
            train_peptides.groupby(["patient_id", "visit_month"])["PeptideAbundance"]
            .count()
            .rename("Pep_count")
            .reset_index()
        )

        # Pre-pivot static training protein & peptide abundance matrices once
        pro_dedup = train_proteins.drop_duplicates(
            subset=["visit_id", "patient_id", "visit_month", "UniProt"]
        )
        self.pro_pivot_base = pro_dedup.pivot(
            index=["visit_id", "patient_id", "visit_month"],
            columns="UniProt",
            values="NPX",
        ).reset_index()

        pep_dedup = train_peptides.copy()
        pep_dedup["pep_key"] = (
            pep_dedup["UniProt"].astype(str) + "_" + pep_dedup["Peptide"].astype(str)
        )
        pep_dedup = pep_dedup.drop_duplicates(
            subset=["visit_id", "patient_id", "visit_month", "pep_key"]
        )
        self.pep_pivot_base = pep_dedup.pivot(
            index=["visit_id", "patient_id", "visit_month"],
            columns="pep_key",
            values="PeptideAbundance",
        ).reset_index()

    @staticmethod
    def _compute_cohort_group(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=["patient_id", "visit_month"]).reset_index(drop=True)
        df["visit_month_diff"] = df.groupby("patient_id")["visit_month"].diff()
        df["visit_month_cummin"] = df.groupby("patient_id")["visit_month_diff"].cummin()

        group = np.where(df["visit_month_cummin"].isna(), 0, 1)
        group = np.where(df["visit_month_cummin"] >= 6, 2, group)
        group = np.where(df["visit_month_cummin"] >= 12, 3, group)
        if "supplemental" in df.columns:
            group = np.where(df["supplemental"] == 1, 0, group)
        df["cohort_group"] = group.astype(np.int8)
        return df

    @staticmethod
    def _matrix_rank_stats(pivot_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        meta_cols = ["visit_id", "patient_id", "visit_month"]
        feat_cols = [c for c in pivot_df.columns if c not in meta_cols]
        pivot_df = pivot_df.sort_values(by=["patient_id", "visit_month"]).reset_index(
            drop=True
        )

        # Vectorized symmetric SMAPE normalization: (x - mean) / (x + mean)
        raw = pivot_df[feat_cols].to_numpy(dtype=np.float64)
        col_means = np.nanmean(raw, axis=0, keepdims=True)
        denom = raw + col_means
        norm = np.divide(
            raw - col_means, denom, out=np.full_like(raw, np.nan), where=(denom != 0)
        )

        # 1. Row sum & patient-level expanding mean rank
        row_sum = np.nansum(norm, axis=1)
        stats_df = pivot_df[meta_cols].copy()
        stats_df[f"{prefix}_sum"] = row_sum
        stats_df[f"{prefix}_sum_cummean"] = stats_df.groupby("patient_id")[
            f"{prefix}_sum"
        ].transform(lambda s: s.expanding().mean())
        stats_df = causal_percentile_rank(
            stats_df, f"{prefix}_sum_cummean", f"{prefix}_sum_cummean_rank"
        )

        # 2. Longitudinal per-patient difference variance & expanding mean rank
        norm_df = pd.DataFrame(norm, columns=feat_cols)
        norm_df["patient_id"] = pivot_df["patient_id"].values
        diff_vals = norm_df.groupby("patient_id")[feat_cols].diff().to_numpy(dtype=np.float64)

        valid_counts = np.sum(~np.isnan(diff_vals), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            diff_var = np.where(valid_counts >= 2, np.nanvar(diff_vals, axis=1, ddof=1), np.nan)

        stats_df[f"{prefix}_diff_var"] = diff_var
        stats_df[f"{prefix}_diff_var_cummean"] = stats_df.groupby("patient_id")[
            f"{prefix}_diff_var"
        ].transform(lambda s: s.expanding().mean())
        stats_df = causal_percentile_rank(
            stats_df, f"{prefix}_diff_var_cummean", f"{prefix}_diff_var_cummean_rank"
        )

        return stats_df[
            meta_cols + [f"{prefix}_sum_cummean_rank", f"{prefix}_diff_var_cummean_rank"]
        ]

    def extract_patient_visit_table(
        self,
        extra_clinical: pd.DataFrame | None = None,
        extra_proteins: pd.DataFrame | None = None,
        extra_peptides: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if extra_clinical is not None and len(extra_clinical) > 0:
            clin = pd.concat(
                [
                    self.base_clinical,
                    extra_clinical[["patient_id", "visit_month", "visit_id"]],
                ],
                ignore_index=True,
            ).drop_duplicates(subset=["patient_id", "visit_month", "visit_id"])
        else:
            clin = self.base_clinical.copy()

        clin = self._compute_cohort_group(clin)

        # Combine richness counts
        if extra_proteins is not None and len(extra_proteins) > 0:
            ext_pro_cnt = (
                extra_proteins.groupby(["patient_id", "visit_month"])["NPX"]
                .count()
                .rename("NPX_count")
                .reset_index()
            )
            pro_cnt = pd.concat([self.pro_counts_base, ext_pro_cnt], ignore_index=True)
            pro_cnt = pro_cnt.drop_duplicates(subset=["patient_id", "visit_month"], keep="last")
        else:
            pro_cnt = self.pro_counts_base.copy()

        if extra_peptides is not None and len(extra_peptides) > 0:
            ext_pep_cnt = (
                extra_peptides.groupby(["patient_id", "visit_month"])["PeptideAbundance"]
                .count()
                .rename("Pep_count")
                .reset_index()
            )
            pep_cnt = pd.concat([self.pep_counts_base, ext_pep_cnt], ignore_index=True)
            pep_cnt = pep_cnt.drop_duplicates(subset=["patient_id", "visit_month"], keep="last")
        else:
            pep_cnt = self.pep_counts_base.copy()

        pro_cnt = pro_cnt.sort_values(by=["patient_id", "visit_month"]).reset_index(drop=True)
        pep_cnt = pep_cnt.sort_values(by=["patient_id", "visit_month"]).reset_index(drop=True)
        pro_cnt = causal_percentile_rank(pro_cnt, "NPX_count", "NPX_count_rank")
        pep_cnt = causal_percentile_rank(pep_cnt, "Pep_count", "Pep_count_rank")

        clin = clin.merge(
            pro_cnt[["patient_id", "visit_month", "NPX_count", "NPX_count_rank"]],
            on=["patient_id", "visit_month"],
            how="left",
        )
        clin = clin.merge(
            pep_cnt[["patient_id", "visit_month", "Pep_count", "Pep_count_rank"]],
            on=["patient_id", "visit_month"],
            how="left",
        )
        clin = clin.sort_values(by=["patient_id", "visit_month"]).reset_index(drop=True)

        clin["NPX_count_rank_cummean"] = (
            clin.groupby("patient_id")["NPX_count_rank"]
            .transform(lambda s: s.expanding().mean())
            .groupby(clin["patient_id"])
            .ffill()
        )
        clin["NPX_count_rank_cummin"] = (
            clin.groupby("patient_id")["NPX_count_rank"]
            .cummin()
            .groupby(clin["patient_id"])
            .ffill()
        )
        clin["Pep_count_rank_cummean"] = (
            clin.groupby("patient_id")["Pep_count_rank"]
            .transform(lambda s: s.expanding().mean())
            .groupby(clin["patient_id"])
            .ffill()
        )
        clin["Pep_count_rank_cummin"] = (
            clin.groupby("patient_id")["Pep_count_rank"]
            .cummin()
            .groupby(clin["patient_id"])
            .ffill()
        )

        # Combine pivoted abundance matrices
        if extra_proteins is not None and len(extra_proteins) > 0:
            ext_p = extra_proteins.drop_duplicates(
                subset=["visit_id", "patient_id", "visit_month", "UniProt"]
            ).pivot(
                index=["visit_id", "patient_id", "visit_month"],
                columns="UniProt",
                values="NPX",
            ).reset_index()
            pro_piv = pd.concat([self.pro_pivot_base, ext_p], ignore_index=True)
            pro_piv = pro_piv.drop_duplicates(
                subset=["visit_id", "patient_id", "visit_month"], keep="last"
            )
        else:
            pro_piv = self.pro_pivot_base

        if extra_peptides is not None and len(extra_peptides) > 0:
            ext_pep = extra_peptides.copy()
            ext_pep["pep_key"] = (
                ext_pep["UniProt"].astype(str) + "_" + ext_pep["Peptide"].astype(str)
            )
            ext_pep = ext_pep.drop_duplicates(
                subset=["visit_id", "patient_id", "visit_month", "pep_key"]
            ).pivot(
                index=["visit_id", "patient_id", "visit_month"],
                columns="pep_key",
                values="PeptideAbundance",
            ).reset_index()
            pep_piv = pd.concat([self.pep_pivot_base, ext_pep], ignore_index=True)
            pep_piv = pep_piv.drop_duplicates(
                subset=["visit_id", "patient_id", "visit_month"], keep="last"
            )
        else:
            pep_piv = self.pep_pivot_base

        pro_stats = self._matrix_rank_stats(pro_piv, "pro")
        pep_stats = self._matrix_rank_stats(pep_piv, "pep")

        clin = clin.merge(pro_stats, on=["visit_id", "patient_id", "visit_month"], how="left")
        clin = clin.merge(pep_stats, on=["visit_id", "patient_id", "visit_month"], how="left")

        for col in (
            "pro_sum_cummean_rank",
            "pro_diff_var_cummean_rank",
            "pep_sum_cummean_rank",
            "pep_diff_var_cummean_rank",
        ):
            clin[col] = clin.groupby("patient_id")[col].ffill()

        # Weighted harmonic mean composite biomarker indices (2:1 protein-to-peptide ratio)
        cnt_cummean_hm = 3.0 / (
            (2.0 / clin["NPX_count_rank_cummean"]) + (1.0 / clin["Pep_count_rank_cummean"])
        )
        cnt_cummin_hm = 3.0 / (
            (2.0 / clin["NPX_count_rank_cummin"]) + (1.0 / clin["Pep_count_rank_cummin"])
        )
        cnt_hm = 0.5 * (cnt_cummean_hm + cnt_cummin_hm)

        sum_rank_hm = 3.0 / (
            (2.0 / clin["pro_sum_cummean_rank"]) + (1.0 / clin["pep_sum_cummean_rank"])
        )
        diff_var_hm = 3.0 / (
            (2.0 / clin["pro_diff_var_cummean_rank"])
            + (1.0 / clin["pep_diff_var_cummean_rank"])
        )
        cnt_and_sum = 2.0 / ((1.0 / cnt_hm) + (1.0 / sum_rank_hm))

        clin["bio_worse_idx"] = np.minimum(cnt_hm, cnt_and_sum).fillna(0.5)
        clin["bio_better_idx"] = (0.5 * (sum_rank_hm + diff_var_hm)).fillna(0.5)

        # CSF sampling compliance indicators at months 0, 6, 12, 18, 24
        csf_cols = []
        for m in CSF_TRACK_MONTHS:
            c_abs = f"csf_m{m}_abs"
            c_pres = f"csf_m{m}_pres"
            clin[c_abs] = np.where(
                clin["NPX_count"].isna() & (clin["visit_month"] == m), 1, 0
            )
            clin[c_pres] = np.where(
                ~clin["NPX_count"].isna() & (clin["visit_month"] == m), 1, 0
            )
            csf_cols.extend([c_abs, c_pres])

        for col in csf_cols:
            clin[col] = clin.groupby("patient_id")[col].cumsum()

        for m in CSF_TRACK_MONTHS:
            c_abs = f"csf_m{m}_abs"
            c_pres = f"csf_m{m}_pres"
            clin[c_abs] = np.where(
                (clin[c_abs] + clin[c_pres] == 0) & (clin["visit_month"] >= m),
                1,
                clin[c_abs],
            )

        keep_cols = ["visit_id", "cohort_group", "bio_worse_idx", "bio_better_idx"] + csf_cols
        return clin[keep_cols].drop_duplicates(subset=["visit_id"], keep="last")


def compute_biomarker_trajectory_predictions(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes:
      1. Full Cohort-Biomarker Trajectory prediction (base curve + molecular/CSF drift)
      2. Pure patient-specific molecular/CSF drift (delta) for calibrating ML predictions
    """
    cg = rows["cohort_group"].to_numpy()
    is_pd = cg <= 2

    f1 = is_pd & (rows["csf_m6_abs"].to_numpy() == 1)
    f2 = (
        is_pd
        & (rows["csf_m6_pres"].to_numpy() == 1)
        & (rows["csf_m12_abs"].to_numpy() == 1)
    )
    f3 = is_pd & (rows["csf_m6_pres"].to_numpy() == 1)
    f4 = (
        is_pd
        & (rows["csf_m6_pres"].to_numpy() == 1)
        & (rows["csf_m12_pres"].to_numpy() == 1)
    )
    f5 = (
        is_pd
        & (rows["csf_m6_abs"].to_numpy() == 1)
        & (rows["csf_m18_pres"].to_numpy() == 1)
    )
    f6 = is_pd & (rows["bio_worse_idx"].to_numpy() >= 0.7)
    f7 = is_pd & (rows["bio_worse_idx"].to_numpy() >= 0.8)
    f7_ctrl = (cg == 3) & (rows["bio_worse_idx"].to_numpy() >= 0.8)
    f8 = is_pd & (rows["bio_better_idx"].to_numpy() <= 0.3)
    f9 = is_pd & (rows["bio_better_idx"].to_numpy() <= 0.2)

    drift_1 = (-1 * f5 + 1 * f7 + 1 * f7_ctrl).astype(np.float64)
    drift_2 = (-1 * f1 - 1 * f2 + 1 * f3 - 1 * f5 + 1 * f6 + 1 * f7 - 1 * f9).astype(
        np.float64
    )
    drift_3 = (
        -2 * f1 + 2 * f3 + 1 * f4 - 8 * f5 + 1 * f6 + 2 * f7 - 1 * f8 - 1 * f9
    ).astype(np.float64)
    drift_4 = np.zeros(len(rows), dtype=np.float64)

    target_idx = rows["target_index"].to_numpy()
    target_months = np.clip(rows["target_month"].to_numpy(), 0, 144)

    base_preds = np.array(
        [
            PROGRESSION_CURVES.get((int(tm), int(c), f"updrs_{int(ti)}"), 0)
            for tm, c, ti in zip(target_months, cg, target_idx)
        ],
        dtype=np.float64,
    )

    drifts = np.select(
        [target_idx == 1, target_idx == 2, target_idx == 3, target_idx == 4],
        [drift_1, drift_2, drift_3, drift_4],
        default=0.0,
    )

    full_bio_preds = np.clip(base_preds + drifts, 0.0, 86.0)
    return full_bio_preds, drifts


def add_features(
    rows: pd.DataFrame,
    proteins: pd.DataFrame,
    source_clinical: pd.DataFrame,
    bio_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = rows.copy()
    clinical_visits = {
        month: set(
            source_clinical.loc[source_clinical["visit_month"] == month, "patient_id"]
        )
        for month in VISIT_MONTHS
    }
    blood_visits = {
        month: set(proteins.loc[proteins["visit_month"] == month, "patient_id"])
        for month in VISIT_MONTHS
    }
    for month in VISIT_MONTHS:
        result[f"visit_{month}m"] = (
            result["patient_id"].isin(clinical_visits[month])
            & (result["visit_month"] >= month)
        ).astype(np.int8)
        result[f"btest_{month}m"] = (
            result["patient_id"].isin(blood_visits[month])
            & (result["visit_month"] >= month)
        ).astype(np.int8)
        result[f"target_month_eq_{month}"] = (result["target_month"] == month).astype(
            np.int8
        )
        result[f"visit_month_eq_{month}"] = (result["visit_month"] == month).astype(
            np.int8
        )
    for horizon in HORIZONS:
        result[f"horizon_eq_{horizon}"] = (result["horizon"] == horizon).astype(np.int8)

    result["horizon_scaled"] = result["horizon"] / 24.0
    result["blood_taken"] = (
        result["visit_id"].isin(set(proteins["visit_id"])).astype(np.int8)
    )
    months_by_patient = source_clinical.groupby("patient_id")["visit_month"].agg(
        lambda values: sorted(set(values))
    )
    result["count_nonannual_visits"] = [
        sum(
            month <= visit_month and month % 12 != 0
            for month in months_by_patient.get(patient, ())
        )
        for patient, visit_month in zip(result["patient_id"], result["visit_month"])
    ]

    if bio_table is not None:
        result = result.merge(bio_table, on="visit_id", how="left")
        result["cohort_group"] = np.where(
            result["supplemental"] == 1,
            0,
            result["cohort_group"].fillna(0),
        ).astype(np.int8)
        result["bio_worse_idx"] = result["bio_worse_idx"].fillna(0.5).astype(np.float32)
        result["bio_better_idx"] = result["bio_better_idx"].fillna(0.5).astype(np.float32)
        for m in CSF_TRACK_MONTHS:
            result[f"csf_m{m}_abs"] = result[f"csf_m{m}_abs"].fillna(0).astype(np.int8)
            result[f"csf_m{m}_pres"] = result[f"csf_m{m}_pres"].fillna(0).astype(np.int8)
    else:
        result["cohort_group"] = np.int8(0)
        result["bio_worse_idx"] = np.float32(0.5)
        result["bio_better_idx"] = np.float32(0.5)
        for m in CSF_TRACK_MONTHS:
            result[f"csf_m{m}_abs"] = np.int8(0)
            result[f"csf_m{m}_pres"] = np.int8(0)

    result["is_annual_control"] = (result["cohort_group"] == 3).astype(np.int8)
    return result


def make_examples(
    source_clinical: pd.DataFrame,
    proteins: pd.DataFrame,
    supplemental: int,
    bio_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_rows = source_clinical.rename(
        columns={"visit_month": "target_month", "visit_id": "target_visit_id"}
    )
    anchor_rows = source_clinical[["patient_id", "visit_month", "visit_id"]]
    paired = target_rows.merge(anchor_rows, on="patient_id", how="inner")
    paired["horizon"] = paired["target_month"] - paired["visit_month"]
    paired = paired[
        paired["horizon"].isin(HORIZONS) & paired["visit_month"].isin(VISIT_MONTHS)
    ].copy()
    paired["supplemental"] = supplemental
    paired = add_features(paired, proteins, source_clinical, bio_table=bio_table)
    examples = []
    for target_index, target_name in enumerate(TARGETS, 1):
        block = paired.copy()
        block["target"] = block[target_name]
        block["target_index"] = target_index
        examples.append(block)
    result = pd.concat(examples, ignore_index=True)
    result = result.dropna(subset=["target"]).copy()
    result["target"] = result["target"].astype(np.int64)
    for target_index in range(1, 5):
        result[f"target_is_{target_index}"] = (
            result["target_index"] == target_index
        ).astype(np.int8)
    return result


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    tabular_tree = [
        "target_index",
        "visit_month",
        "horizon",
        "target_month",
        "blood_taken",
        "supplemental",
        "visit_6m",
        "visit_18m",
        "visit_48m",
        "count_nonannual_visits",
        "cohort_group",
        "bio_worse_idx",
        "bio_better_idx",
    ]
    tabular_nn = [
        "visit_6m",
        "visit_18m",
        "visit_48m",
        "supplemental",
        "horizon_scaled",
        "is_annual_control",
        "bio_worse_idx",
        "bio_better_idx",
    ] + [
        column
        for column in frame
        if column.startswith(
            ("target_month_eq_", "visit_month_eq_", "horizon_eq_", "target_is_")
        )
    ]
    return {
        "tabular_tree": tabular_tree,
        "tabular_nn": list(dict.fromkeys(tabular_nn)),
    }


class TabularRegressor(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.LeakyReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(1)


def smape_scaled_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(target - prediction)
        / (torch.abs(0.01 + target) + torch.abs(0.01 + prediction))
    ).mean()


def prepare_training(
    data_dir: Path,
) -> tuple[pd.DataFrame, BiomarkerTrajectoryEngine]:
    clinical = pd.read_csv(data_dir / "train_clinical_data.csv")
    supplemental = pd.read_csv(data_dir / "supplemental_clinical_data.csv")
    # Filter abnormal short-followup supplemental records (< 6 months max)
    sup_max_month = supplemental.groupby("patient_id")["visit_month"].transform("max")
    supplemental = supplemental[sup_max_month >= 6].copy()
    supplemental.loc[supplemental["visit_month"] == 5, "visit_month"] = 6

    proteins = pd.read_csv(data_dir / "train_proteins.csv")
    peptides = pd.read_csv(data_dir / "train_peptides.csv")

    bio_engine = BiomarkerTrajectoryEngine(clinical, proteins, peptides)
    train_bio_table = bio_engine.extract_patient_visit_table()

    clinical_examples = make_examples(
        clinical, proteins, supplemental=0, bio_table=train_bio_table
    )
    supplemental_examples = make_examples(
        supplemental, proteins, supplemental=1, bio_table=None
    )
    examples = pd.concat(
        [clinical_examples, supplemental_examples], ignore_index=True
    )
    return examples, bio_engine


def train_lgbm(examples: pd.DataFrame, features: list[str], seed: int) -> Any:
    model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="multiclass",
        num_class=87,
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=208,
        max_depth=14,
        min_child_samples=850,
        colsample_bytree=0.52,
        reg_alpha=1e-7,
        reg_lambda=0.15,
        max_bin=240,
        random_state=seed,
        verbosity=-1,
        n_jobs=4,
        force_col_wise=True,
    )
    model.fit(
        examples[features], examples["target"], categorical_feature=["target_index"]
    )
    return model


def train_mlp(examples: pd.DataFrame, features: list[str], seed: int) -> TabularRegressor:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    values = torch.tensor(examples[features].to_numpy(np.float32))
    targets = torch.tensor(examples["target"].to_numpy(np.float32) / 100.0)
    loader = DataLoader(
        TensorDataset(values, targets),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TabularRegressor(len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    total_steps = 10 * len(loader)
    step_number = 0
    for _ in range(10):
        model.train()
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = smape_scaled_loss(model(batch), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_number += 1
            scale = 0.5 * (1.0 + np.cos(np.pi * step_number / total_steps))
            for group in optimizer.param_groups:
                group["lr"] = 0.002 * scale
    model.eval()
    return model


def add_multiview_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.reset_index(drop=True).copy()
    base = []
    for month in VISIT_MONTHS:
        name = f"have_{month}"
        result[name] = np.where(
            result.visit_month < month, -1, result[f"visit_{month}m"]
        ).astype(np.float32)
        base.append(name)
    for month in VISIT_MONTHS:
        name = f"anchor_{month}"
        result[name] = (result.visit_month == month).astype(np.float32)
        base.append(name)
    result["supplemental_feature"] = result.supplemental.astype(np.float32)
    result["annual_control_feature"] = result.is_annual_control.astype(np.float32)
    result["bio_worse_feature"] = result.bio_worse_idx.astype(np.float32)
    result["bio_better_feature"] = result.bio_better_idx.astype(np.float32)
    base.extend(
        [
            "supplemental_feature",
            "annual_control_feature",
            "bio_worse_feature",
            "bio_better_feature",
        ]
    )
    return result, base


def structured_layout(
    frame: pd.DataFrame, kind: str, include_target: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    rows, base = add_multiview_features(frame)
    if kind == "scalar":
        features = (
            base
            + [f"horizon_eq_{h}" for h in HORIZONS]
            + [f"target_is_{target}" for target in range(1, 5)]
        )
        group = np.arange(len(rows))
        output = np.zeros(len(rows), dtype=int)
        x = rows[features].to_numpy(np.float32)
        y = rows.target.to_numpy(np.float32)[:, None] if include_target else None
        return x, y, group, output
    if kind == "target4":
        keys = ["patient_id", "visit_month", "target_month", "horizon", "supplemental"]
        features = base + [f"horizon_eq_{h}" for h in HORIZONS]
        output = rows.target_index.to_numpy() - 1
        width = 4
    elif kind == "horizon4":
        keys = ["patient_id", "visit_month", "target_index", "supplemental"]
        features = base + [f"target_is_{target}" for target in range(1, 5)]
        output = rows.horizon.map({h: i for i, h in enumerate(HORIZONS)}).to_numpy()
        width = 4
    elif kind == "joint16":
        keys = ["patient_id", "visit_month", "supplemental"]
        features = base
        h_index = rows.horizon.map({h: i for i, h in enumerate(HORIZONS)}).to_numpy()
        output = (rows.target_index.to_numpy() - 1) * 4 + h_index
        width = 16
    else:
        raise ValueError(kind)

    group = rows.groupby(keys, sort=False).ngroup().to_numpy()
    first = rows.groupby(group, sort=True).first()
    x = first[features].to_numpy(np.float32)
    y = None
    if include_target:
        y = np.full((len(first), width), np.nan, dtype=np.float32)
        y[group, output] = rows.target.to_numpy(np.float32)
    return x, y, group, output


class MultiViewNet(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, outputs),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def fit_multiview(examples: pd.DataFrame, kind: str, seed: int) -> MultiViewNet:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y, _, _ = structured_layout(examples, kind, include_target=True)
    assert y is not None
    model = MultiViewNet(x.shape[1], y.shape[1])
    output_layer = model.layers[-2]
    nn.init.normal_(output_layer.weight, mean=0.0, std=1e-3)
    output_layer.bias.data.copy_(
        torch.tensor(
            np.nan_to_num(np.nanmedian(y / 100.0, axis=0)), dtype=torch.float32
        )
    )
    loader = DataLoader(
        TensorDataset(torch.tensor(x), torch.tensor(y / 100.0)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    for _ in range(4):
        model.train()
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            mask = torch.isfinite(target)
            clean = torch.nan_to_num(target)
            loss = torch.abs(model(batch) - clean)[mask].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    for _ in range(12):
        model.train()
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            mask = torch.isfinite(target)
            clean = torch.nan_to_num(target)
            loss = (
                torch.abs(prediction - clean)
                / (torch.abs(0.01 + prediction) + torch.abs(0.01 + clean))
            )[mask].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


def predict_multiview(
    rows: pd.DataFrame, models: dict[str, list[MultiViewNet]]
) -> np.ndarray:
    kind_predictions = []
    with torch.inference_mode():
        for kind in MULTIVIEW_KINDS:
            x, _, group, output = structured_layout(rows, kind, include_target=False)
            values = torch.tensor(x)
            bags = [model(values).numpy() * 100.0 for model in models[kind]]
            matrix = np.mean(bags, axis=0)
            kind_predictions.append(matrix[group, output])
    return np.clip(np.mean(kind_predictions, axis=0), 0.0, 86.0)


def parse_submission_rows(sample_submission: pd.DataFrame) -> pd.DataFrame:
    parsed = sample_submission["prediction_id"].str.extract(PREDICTION_ID_PATTERN)
    if parsed.isna().any().any():
        bad = (
            sample_submission.loc[parsed.isna().any(axis=1), "prediction_id"]
            .head()
            .tolist()
        )
        raise ValueError(f"Unexpected prediction_id format: {bad}")
    rows = pd.DataFrame(
        {
            "patient_id": parsed["patient"].astype(int),
            "visit_month": parsed["visit"].astype(int),
            "target_index": parsed["target"].astype(int),
            "horizon": parsed["horizon"].astype(int),
        }
    )
    rows["target_month"] = rows["visit_month"] + rows["horizon"]
    rows["visit_id"] = (
        rows["patient_id"].astype(str) + "_" + rows["visit_month"].astype(str)
    )
    rows["supplemental"] = 0
    for target_index in range(1, 5):
        rows[f"target_is_{target_index}"] = (
            rows["target_index"] == target_index
        ).astype(np.int8)
    return rows


def run_submission(data_dir: Path, environment: Any) -> None:
    torch.set_num_threads(4)
    examples, bio_engine = prepare_training(data_dir)
    features = feature_sets(examples)
    tree_features, nn_features = features["tabular_tree"], features["tabular_nn"]

    trees = [train_lgbm(examples, tree_features, seed) for seed in TREE_SEEDS]
    networks = [train_mlp(examples, nn_features, seed) for seed in TABULAR_NN_SEEDS]
    multiview_models = {
        kind: [fit_multiview(examples, kind, seed) for seed in MULTIVIEW_SEEDS]
        for kind in MULTIVIEW_KINDS
    }

    observed_clinical: list[pd.DataFrame] = []
    observed_proteins: list[pd.DataFrame] = []
    observed_peptides: list[pd.DataFrame] = []

    for clinical, peptides, proteins, sample in environment.iter_test():
        observed_clinical.append(
            clinical[["patient_id", "visit_month", "visit_id"]].drop_duplicates()
        )
        observed_proteins.append(proteins)
        observed_peptides.append(peptides)

        all_test_clin = pd.concat(observed_clinical, ignore_index=True)
        all_test_pro = pd.concat(observed_proteins, ignore_index=True)
        all_test_pep = pd.concat(observed_peptides, ignore_index=True)

        # 1. Extract causal biomarker & cohort trajectory table
        bio_table = bio_engine.extract_patient_visit_table(
            extra_clinical=all_test_clin,
            extra_proteins=all_test_pro,
            extra_peptides=all_test_pep,
        )

        # 2. Build unified test features
        parsed_rows = parse_submission_rows(sample)
        rows = add_features(
            parsed_rows,
            all_test_pro[["patient_id", "visit_month", "visit_id"]].drop_duplicates(),
            all_test_clin[["patient_id", "visit_month"]].drop_duplicates(),
            bio_table=bio_table,
        )

        # 3. Branch A: Cohort-Biomarker Trajectory prediction & molecular residual drift
        bio_trajectory_pred, bio_drift = compute_biomarker_trajectory_predictions(rows)

        # 4. Branch B: Tabular GBDT + Tabular MLP + 4-View Multi-Structure Neural Ensemble
        probabilities = np.zeros((len(rows), 87), dtype=np.float64)
        for tree in trees:
            probabilities[:, tree.classes_.astype(int)] += tree.predict_proba(
                rows[tree_features]
            ) / len(trees)
        tree_prediction = decode_minimum_risk(probabilities)

        values = torch.tensor(rows[nn_features].to_numpy(np.float32))
        with torch.inference_mode():
            nn_prediction = np.mean(
                [model(values).numpy() * 100.0 for model in networks], axis=0
            )
        nn_prediction = np.clip(nn_prediction, 0.0, 86.0)
        tabular_consensus = 0.5 * (tree_prediction + nn_prediction)

        multiview_prediction = predict_multiview(rows, multiview_models)
        mv_weights = rows["target_index"].map(TARGET_MULTIVIEW_WEIGHT).to_numpy()
        ml_ensemble_pred = (
            1.0 - mv_weights
        ) * tabular_consensus + mv_weights * multiview_prediction

        # Apply molecular & CSF-schedule residual drift to calibrate ML ensemble
        ml_calibrated_pred = np.clip(ml_ensemble_pred + 0.5 * bio_drift, 0.0, 86.0)

        # 5. Cohort-aware & target-aware fusion
        bio_weights = rows["target_index"].map(TARGET_BIO_TRAJECTORY_WEIGHT).to_numpy()
        blended = (
            bio_weights * bio_trajectory_pred
            + (1.0 - bio_weights) * ml_calibrated_pred
        )

        # Strictly lock healthy annual control cohort (cohort_group == 3) to control trajectory
        is_healthy_ctrl = rows["cohort_group"].to_numpy() == 3
        final_prediction = np.where(is_healthy_ctrl, bio_trajectory_pred, blended)
        final_prediction = np.clip(np.rint(final_prediction), 0, 86).astype(int)

        submission = sample[["prediction_id"]].copy()
        submission["rating"] = final_prediction
        environment.predict(submission)


def run_kaggle() -> None:
    data_dir = Path("/kaggle/input/amp-parkinsons-disease-progression-prediction")
    sys.path.insert(0, str(data_dir))
    try:
        import amp_pd_peptide_310 as amp_pd_peptide
    except ImportError:
        import amp_pd_peptide
    if hasattr(amp_pd_peptide.make_env, "func_dict"):
        amp_pd_peptide.make_env.func_dict["__called__"] = False
    run_submission(data_dir, amp_pd_peptide.make_env())


if __name__ == "__main__":
    run_kaggle()