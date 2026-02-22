import argparse
import json
import math
import os
import random
from typing import Dict, List
import pandas as pd

Z_TABLE = {
    0.80: 1.28155,
    0.85: 1.43953,
    0.90: 1.64485,
    0.95: 1.95996,
    0.99: 2.57583,
}

def calc_sample_size(
    population_size: int,
    confidence_level: float = 0.90,
    tolerable_error: float = 0.10,
    assumed_p: float = 0.50,
    use_fpc: bool = True,
    round_up: bool = True,
) -> Dict:
    """
    Common attribute-sampling / proportion-style planning formula:

      n0 = (Z^2 * p * (1-p)) / E^2
      if FPC: n = (N * n0) / (N + n0 - 1)

    Notes:
    - assumed_p=0.5 is conservative (max variance).
    - This is a generic planning approach; adapt to your firm's methodology as needed.
    """

    if confidence_level not in Z_TABLE:
        raise ValueError(f"Unsupported confidence_level {confidence_level}. Add to Z_TABLE if needed.")
    z = Z_TABLE[confidence_level]
    e = tolerable_error
    p = assumed_p

    n0 = (z**2 * p * (1 - p)) / (e**2)

    if use_fpc:
        _n = population_size
        n = (_n * n0) / (_n + n0 - 1)
    else:
        n = n0

    final_n = math.ceil(n) if round_up else int(round(n))

    return {
        "confidence_level": confidence_level,
        "z_value": z,
        "tolerable_error": e,
        "assumed_p": p,
        "n0_unbounded": n0,
        "population_size": population_size,
        "finite_population_corrected_n": n if use_fpc else None,
        "final_sample_size": final_n,
        "used_fpc": use_fpc,
        "rounding": "ceil" if round_up else "round",
        "formula_notes": {
            "n0": "(Z^2 * p * (1-p)) / E^2",
            "fpc": "(N * n0) / (N + n0 - 1)"
        }
    }

# Generator: population tab
# I need to modify these names... maybe


METRIC_NAMES = [
    "KYC Exceptions Rate",
    "SAR Filing Timeliness",
    "High-Risk Customer Count",
    "PEP Screening Hits",
    "Transaction Monitoring Alerts",
    "Alert Clearance SLA Breaches",
    "Sanctions Screening Matches",
    "False Positive Rate",
    "EDD Completion Rate",
    "Case Backlog Volume",
    "STR Quality Score",
    "Adverse Media Hits",
]

REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
OWNERS = ["AFC Ops", "Compliance", "Risk Analytics", "AML Investigations"]

def make_population_df(rng: random.Random, n_metrics: int) -> pd.DataFrame:
    n_metrics = max(12, n_metrics)
    metrics = rng.sample(METRIC_NAMES, k=min(len(METRIC_NAMES), n_metrics))
    # if n_metrics > list length, add synthetic names
    while len(metrics) < n_metrics:
        metrics.append(f"Custom Metric {len(metrics)+1}")

    rows = []
    for m in metrics:
        region = rng.choice(REGIONS)
        owner = rng.choice(OWNERS)

        # generate Q2 and Q3 values with plausible drift
        base = rng.uniform(0, 100)
        q2 = max(0, int(base + rng.uniform(-10, 10)))
        q3 = max(0, int(base + rng.uniform(-10, 10)))

        # occasionally make them counts
        is_count = rng.random() < 0.35
        if is_count:
            q2 = int(rng.uniform(0, 5000))
            q3 = max(0, int(q2 + rng.uniform(-800, 800)))

        unit = "count" if is_count else rng.choice(["%", "score", "rate"])
        reported = rng.choice(["Reported", "Calculated", "Estimated"])

        rows.append({
            "Metric": m,
            "Region": region,
            "Owner": owner,
            "Unit": unit,
            "Q2": round(q2, 2) if not is_count else q2,
            "Q3": round(q3, 2) if not is_count else q3,
            "SourceType": reported,
        })

    return pd.DataFrame(rows)

# Serialization for fine-tuning

def df_to_compact_csv(df: pd.DataFrame, max_rows: int = 200) -> str:
    df2 = df.head(max_rows)
    return df2.to_csv(index=False)

def build_prompt(pop_csv: str, confidence: float, tol_err: float) -> str:
    # Keep prompts consistent + machine-readable.
    return (
        "You are an auditor. Using the Population table below, calculate the required sample size "
        f"for audit testing based on a {int(confidence*100)}% confidence level and a {int(tol_err*100)}% tolerable error rate.\n\n"
        "Population (CSV):\n"
        f"{pop_csv}\n\n"
        "Return ONLY valid JSON for a second tab titled 'Sample Size Calculation' with workings and the final sample size."
    )

def build_response(gt: Dict) -> str:
    return json.dumps({"Sample Size Calculation": gt}, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="audit_synth")
    ap.add_argument("--n_examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    # knobs for variability
    ap.add_argument("--min_metrics", type=int, default=40)
    ap.add_argument("--max_metrics", type=int, default=220)
    ap.add_argument("--confidence", type=float, default=0.90)
    ap.add_argument("--tolerable_error", type=float, default=0.10)

    # output formats
    ap.add_argument("--write_sft_jsonl", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    sft_rows: List[Dict[str, str]] = []

    for i in range(args.n_examples):
        ex_id = f"ex_{i:05d}"
        ex_dir = os.path.join(args.out_dir, ex_id)
        os.makedirs(ex_dir, exist_ok=True)

        n_metrics = rng.randint(args.min_metrics, args.max_metrics)
        df = make_population_df(rng, n_metrics=n_metrics)

        # "population size" for sampling: usually the number of items you sample from.
        # Here we treat each metric as an item. If in your real work "population"
        # is something else (e.g., transactions), compute N accordingly.
        pop_size = len(df)

        gt = calc_sample_size(
            population_size=pop_size,
            confidence_level=args.confidence,
            tolerable_error=args.tolerable_error,
            assumed_p=0.50,
            use_fpc=True,
            round_up=True,
        )

        pop_path = os.path.join(ex_dir, "population.csv")
        gt_path = os.path.join(ex_dir, "sample_size_calculation.json")

        df.to_csv(pop_path, index=False)
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

        if args.write_sft_jsonl:
            pop_csv = df_to_compact_csv(df, max_rows=250)
            prompt = build_prompt(pop_csv, args.confidence, args.tolerable_error)
            response = build_response(gt)
            sft_rows.append({"prompt": prompt, "response": response})

    if args.write_sft_jsonl:
        sft_path = os.path.join(args.out_dir, "sft.jsonl")
        with open(sft_path, "w", encoding="utf-8") as f:
            for r in sft_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {args.n_examples} examples to {args.out_dir}")


if __name__ == "__main__":
    main()
