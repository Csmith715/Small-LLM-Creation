"""
This script generates 400 synthetic Anti-Financial Crime Risk Metrics CSV files from an input Excel,
and (within EACH output CSV) selects an audit sample by marking "1" in column K (11th column) per the rules.

Usage:
  python Sample-Select/synthesize_metrics.py --input "data/Sample%20v2.xlsx" --outdir "synthetic_select_data" --n_files 400 --sample_size 25
"""

from __future__ import annotations
import argparse
import os
from typing import Set
import numpy as np
import pandas as pd
from utils import calc_sample_size

# Generation Configuration

SPECIAL_LEGAL_ENTITIES = [
    "CB Cash Italy",
    "CB Correspondent Banking Greece",
    "IB Debt Markets Luxembourg",
    "CB Trade Finance Brazil",
    "PB EMEA UAE",
]

SPECIAL_ENTITY_ROWS = [
    # (Legal Entity, Division, Sub-Division, Country)
    ("CB Cash Italy", "Corporate Bank", "Cash", "Italy"),
    ("CB Correspondent Banking Greece", "Corporate Bank", "Correspondent Banking", "Greece"),
    ("IB Debt Markets Luxembourg", "Markets", "Debt Markets", "Luxembourg"),
    ("CB Trade Finance Brazil", "Corporate Bank", "Trade Finance", "Brazil"),
    ("PB EMEA UAE", "Private Bank", "EMEA", "UAE"),
]

HIGH_RISK_METRICS = ["A1", "C1"]
FOCUS_COUNTRIES = {"Cayman Islands", "Pakistan", "UAE"}
FOCUS_SUBDIVS = {"Trade Finance", "Correspondent Banking"}

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def compute_variance_pct(q3: pd.Series, q2: pd.Series) -> pd.Series:
    """
    Match the input file's behavior: variance = ((Q3 - Q2) / Q2) * 100 when Q2 != 0, else 0.
    """
    q3n = _coerce_numeric(q3)
    q2n = _coerce_numeric(q2)
    out = np.zeros(len(q2n), dtype=float)
    mask = q2n != 0
    out[mask] = ((q3n[mask] - q2n[mask]) / q2n[mask]) * 100.0
    return pd.Series(out, index=q3.index)


def percent_change_abs(q3: pd.Series, q2: pd.Series) -> pd.Series:
    """
    For selection logic: absolute % change. If Q2==0 and Q3!=0 then treat as very large.
    """
    q3n = _coerce_numeric(q3).astype(float)
    q2n = _coerce_numeric(q2).astype(float)

    out = np.zeros(len(q2n), dtype=float)
    nonzero = q2n != 0
    out[nonzero] = np.abs((q3n[nonzero] - q2n[nonzero]) / q2n[nonzero]) * 100.0

    # If Q2 == 0:
    zero = ~nonzero
    out[zero] = np.where(q3n[zero] == 0, 0.0, 1_000_000.0)  # "exceptionally large"
    return pd.Series(out, index=q3.index)


def ensure_column_k_sample_selected(df: pd.DataFrame, sample_col_name: str = "Sample Selected") -> pd.DataFrame:
    """
    Ensure the sampling indicator lives in column K (11th column).
    If the input has only 10 columns, insert a blank column before the sample column.
    If the sample column doesn't exist, create it as the last column, then pad to 11 columns.
    """
    df = df.copy()

    if sample_col_name not in df.columns:
        df[sample_col_name] = ""

    # Move sample column to end for easier padding
    cols = [c for c in df.columns if c != sample_col_name] + [sample_col_name]
    df = df[cols]

    # Pad with blank columns until we have at least 11 columns,
    # keeping the sample column as the last column (K).
    while df.shape[1] < 11:
        # Insert a blank column right before the last column (sample column)
        insert_at = df.shape[1] - 1
        new_col = f"Blank_{df.shape[1]+1}"
        left = list(df.columns[:insert_at])
        right = list(df.columns[insert_at:])
        df[new_col] = ""
        df = df[left + [new_col] + right]

    # If we have > 11 columns, we still keep sample column present; column lettering becomes ambiguous,
    # but we keep "Sample Selected" at the end to match "column K" expectation in most templates.
    return df


def perturb_values(base: pd.DataFrame, q3_col: str, q2_col: str, variance_col: str, seed: int) -> pd.DataFrame:
    """
    Create synthetic but similar numeric behavior:
    - Small multiplicative noise across most rows
    - Some rows get large shifts to create high-variance cases
    - Some rows forced to both-zero
    """
    rng = np.random.default_rng(seed)
    df = base.copy()

    q2 = _coerce_numeric(df[q2_col]).astype(float).to_numpy()
    q3 = _coerce_numeric(df[q3_col]).astype(float).to_numpy()

    # Small noise
    noise_q2 = rng.lognormal(mean=0.0, sigma=0.08, size=len(df))
    noise_q3 = rng.lognormal(mean=0.0, sigma=0.08, size=len(df))
    q2 = np.round(q2 * noise_q2).astype(float)
    q3 = np.round(q3 * noise_q3).astype(float)

    # Inject some big swings (emphasize exceptionally large percentage changes)
    n_big = max(10, int(0.05 * len(df)))
    big_idx = rng.choice(len(df), size=n_big, replace=False)
    big_mult = rng.uniform(1.8, 8.0, size=n_big)  # large changes
    direction = rng.choice([-1, 1], size=n_big)

    # If direction == 1 => boost Q3; if -1 => reduce Q3 (but keep non-negative)
    q3_big = q3[big_idx] * np.where(direction > 0, big_mult, 1.0 / big_mult)
    q3[big_idx] = np.maximum(0, np.round(q3_big))

    # Force some both-zero rows
    n_zero_pairs = max(5, int(0.01 * len(df)))
    zero_idx = rng.choice(len(df), size=n_zero_pairs, replace=False)
    q2[zero_idx] = 0
    q3[zero_idx] = 0

    df[q2_col] = q2.astype(int)
    df[q3_col] = q3.astype(int)
    df[variance_col] = compute_variance_pct(df[q3_col], df[q2_col])

    return df


def inject_required_entities_and_metrics(
    df: pd.DataFrame,
    q3_col: str,
    q2_col: str,
    variance_col: str,
    seed: int,
) -> pd.DataFrame:
    """
    Ensure the presence of:
      - the 5 specified Legal Entities
      - metrics A1 and C1
      - Trade Finance & Correspondent Banking rows (as Sub-Division entries)
    Strategy: overwrite a small number of existing rows (keeps row count stable).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    # Choose 5 distinct rows to overwrite with special entities
    idxs = rng.choice(len(out), size=len(SPECIAL_ENTITY_ROWS), replace=False)
    for i, (le, div, subdiv, ctry) in enumerate(SPECIAL_ENTITY_ROWS):
        idx = int(idxs[i])
        out.loc[idx, "Legal Entity"] = le
        out.loc[idx, "Division"] = div
        out.loc[idx, "Sub-Division"] = subdiv
        out.loc[idx, "Country"] = ctry

        # Create a high-variance pattern on these to make them easy to sample if needed
        base_val = int(rng.integers(50, 5000))
        out.loc[idx, q2_col] = base_val
        out.loc[idx, q3_col] = int(np.round(base_val * rng.uniform(1.4, 6.0)))

    # Inject A1 & C1 by overwriting two rows' KRI label
    idxs2 = rng.choice(len(out), size=2, replace=False)
    out.loc[int(idxs2[0]), "KRIs"] = "A1"
    out.loc[int(idxs2[1]), "KRIs"] = "C1"

    # Recompute variance after edits
    out[variance_col] = compute_variance_pct(out[q3_col], out[q2_col])
    return out


def select_audit_sample(
    df: pd.DataFrame,
    q3_col: str,
    q2_col: str,
    variance_col: str,
    sample_col: str,
    seed: int,
) -> pd.DataFrame:
    """
    Mark an audit sample with "1" such that:
      i) each selected row satisfies at least one of the listed criteria (excluding the overall coverage bullet),
      ii) across selected rows, each listed criterion is satisfied at least once,
      plus "Ensure coverage across all Divisions and sub-Divisions."
    """
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.head(np.random.randint(25,100))
    out = df.copy()
    out[sample_col] = ""
    sample_size = calc_sample_size(out.shape[0])

    # Criterion flags per-row (excluding the set-level division/subdivision coverage bullet)
    pct_abs = percent_change_abs(out[q3_col], out[q2_col])

    crit_high_variance = pct_abs > 20.0
    crit_special_entity = out["Legal Entity"].isin(SPECIAL_LEGAL_ENTITIES)
    crit_high_risk_metric = out["KRIs"].isin(HIGH_RISK_METRICS)
    crit_both_zero = (_coerce_numeric(out[q2_col]) == 0) & (_coerce_numeric(out[q3_col]) == 0)
    crit_focus_business = out["Sub-Division"].astype(str).isin(FOCUS_SUBDIVS)
    crit_focus_country = out["Country"].astype(str).isin(FOCUS_COUNTRIES)

    any_row_criterion = (
        crit_high_variance
        | crit_special_entity
        | crit_high_risk_metric
        | crit_both_zero
        | crit_focus_business
        | crit_focus_country
    )

    # Start selecting required coverage for each criterion
    selected: Set[int] = set()

    def pick_one(data_mask: pd.Series) -> None:
        idxs = out.index[data_mask].to_list()
        if not idxs:
            return
        selected.add(int(rng.choice(idxs)))

    # Ensure each criterion appears at least once
    pick_one(crit_high_variance)
    pick_one(crit_special_entity)
    pick_one(crit_high_risk_metric)
    pick_one(crit_both_zero)
    pick_one(crit_focus_business)
    pick_one(crit_focus_country)

    # Ensure coverage across all Divisions and Sub-Divisions
    divisions = out["Division"].astype(str).unique().tolist()
    subdivs = out["Sub-Division"].astype(str).unique().tolist()

    # Prefer rows that ALSO meet at least one main criterion, so every selected row is justified.
    for div in divisions:
        mask = (out["Division"].astype(str) == div) & any_row_criterion
        if mask.any():
            pick_one(mask)
        else:
            # If none in this division meet any criterion, force-create one by taking any row in div and making it high variance
            div_idxs = out.index[out["Division"].astype(str) == div].to_list()
            if div_idxs:
                idx = int(rng.choice(div_idxs))
                # Force high variance via Q3 shock
                q2 = int(_coerce_numeric(out.loc[[idx], q2_col]).iloc[0])
                q3 = int(_coerce_numeric(out.loc[[idx], q3_col]).iloc[0])
                if q2 == 0:
                    q2 = int(rng.integers(10, 500))
                    out.loc[idx, q2_col] = q2
                out.loc[idx, q3_col] = int(np.round(q2 * rng.uniform(1.4, 10.0)))
                selected.add(idx)

    for sd in subdivs:
        mask = (out["Sub-Division"].astype(str) == sd) & any_row_criterion
        if mask.any():
            pick_one(mask)
        else:
            sd_idxs = out.index[out["Sub-Division"].astype(str) == sd].to_list()
            if sd_idxs:
                idx = int(rng.choice(sd_idxs))
                q2 = int(_coerce_numeric(out.loc[[idx], q2_col]).iloc[0])
                if q2 == 0:
                    q2 = int(rng.integers(10, 500))
                    out.loc[idx, q2_col] = q2
                out.loc[idx, q3_col] = int(np.round(q2 * rng.uniform(1.4, 10.0)))
                selected.add(idx)

    # Recompute variance after any forced edits
    out[variance_col] = compute_variance_pct(out[q3_col], out[q2_col])
    pct_abs = percent_change_abs(out[q3_col], out[q2_col])
    crit_high_variance = pct_abs > 20.0
    any_row_criterion = (
        crit_high_variance
        | out["Legal Entity"].isin(SPECIAL_LEGAL_ENTITIES)
        | out["KRIs"].isin(HIGH_RISK_METRICS)
        | ((_coerce_numeric(out[q2_col]) == 0) & (_coerce_numeric(out[q3_col]) == 0))
        | out["Sub-Division"].astype(str).isin(FOCUS_SUBDIVS)
        | out["Country"].astype(str).isin(FOCUS_COUNTRIES)
    )

    # If we need more rows, fill with the highest absolute % variance (emphasize big changes),
    # but ensure every added row satisfies at least one criterion.
    if len(selected) < sample_size:
        candidates = out.index[any_row_criterion].difference(pd.Index(list(selected)))
        # sort candidates by pct_abs desc
        cand_sorted = sorted(
            [int(i) for i in candidates],
            key=lambda i: float(pct_abs.loc[i]),
            reverse=True,
        )
        for idx in cand_sorted:
            if len(selected) >= sample_size:
                break
            selected.add(idx)

    # If still short (rare), relax to any rows with >20% variance after forcing some.
    if len(selected) < sample_size:
        remaining = out.index.difference(pd.Index(list(selected))).to_list()
        rng.shuffle(remaining)
        for idx in remaining:
            if len(selected) >= sample_size:
                break
            # force this row to satisfy high variance so it remains justified
            q2 = int(_coerce_numeric(out.loc[[idx], q2_col]).iloc[0])
            if q2 == 0:
                q2 = int(rng.integers(10, 500))
                out.loc[idx, q2_col] = q2
            out.loc[idx, q3_col] = int(np.round(q2 * rng.uniform(1.4, 10.0)))
            selected.add(int(idx))

    # Final: write sample flags as string "1"
    out.loc[list(selected), sample_col] = "1"

    # Safety: ensure each selected row meets at least one criterion (excluding the set-level coverage bullet)
    # If not, force it to high variance
    selected_list = list(selected)
    pct_abs = percent_change_abs(out[q3_col], out[q2_col])
    for idx in selected_list:
        ok = bool(any_row_criterion.loc[idx])
        if not ok:
            q2 = int(_coerce_numeric(out.loc[[idx], q2_col]).iloc[0])
            if q2 == 0:
                q2 = int(rng.integers(10, 500))
                out.loc[idx, q2_col] = q2
            out.loc[idx, q3_col] = int(np.round(q2 * rng.uniform(1.4, 10.0)))
            out.loc[idx, sample_col] = "1"

    out[variance_col] = compute_variance_pct(out[q3_col], out[q2_col])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input Excel (.xlsx) file")
    parser.add_argument("--outdir", required=True, help="Output directory for CSV files")
    parser.add_argument("--n_files", type=int, default=400, help="Number of synthetic CSV files to generate")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = pd.read_excel(args.input)

    # Expected columns from your sample
    q3_col = "Q3 2024 KRI"
    q2_col = "Q2 2024 KRI"
    variance_col = "Variance"
    sample_col = "Sample Selected"

    # Basic validation
    for col in ["Division", "Sub-Division", "Country", "Legal Entity", "KRIs", q3_col, q2_col, variance_col, sample_col]:
        if col not in base.columns:
            raise ValueError(f"Missing required column: {col}")

    for i in range(1, args.n_files + 1):
        file_seed = args.seed + i * 10_000

        synth = perturb_values(base, q3_col=q3_col, q2_col=q2_col, variance_col=variance_col, seed=file_seed)
        synth = inject_required_entities_and_metrics(synth, q3_col=q3_col, q2_col=q2_col, variance_col=variance_col, seed=file_seed + 7)
        synth = select_audit_sample(
            synth,
            q3_col=q3_col,
            q2_col=q2_col,
            variance_col=variance_col,
            sample_col=sample_col,
            seed=file_seed + 13,
        )

        # Ensure the sample indicator is in column K (11th col) by padding as needed
        synth = ensure_column_k_sample_selected(synth, sample_col_name=sample_col)

        out_path = os.path.join(args.outdir, f"synthetic_{i:03d}.csv")
        synth.to_csv(out_path, index=False)

    print(f"Done. Wrote {args.n_files} CSV files to: {args.outdir}")


if __name__ == "__main__":
    main()
