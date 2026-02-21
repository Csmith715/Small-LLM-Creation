import argparse
import pandas as pd
import json
from pathlib import Path
from utils import calc_sample_size


def find_csv_data(root_dir):
    list_of_dataframes = []
    path = Path(root_dir)
    csv_files = path.glob('**/*.csv')
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df.drop('Blank_11', axis=1)
        list_of_dataframes.append(df)
    return list_of_dataframes


def make_example(synth_df: pd.DataFrame) -> dict:
    sample_size = calc_sample_size(synth_df.shape[0])
    ss_resp = synth_df['Sample Selected'].to_markdown(index=False)
    sdf = synth_df.to_markdown(index=False)
    sys = f"""
    You are an expert data analyst. Given data from a tabular dataset in a Markdown format, you will be asked to perform various statistical analyses. 
    Select a sample of {sample_size} records for audit testing based on the following criteria and indicate sampled rows by entering "1" in a new column, "Sample Selected".
    For the rows not selected for audit testing, enter "" in the Sample Selected column. 
    Ensure that:
    i) each sample selected satisfies at least one criteria listed below, and 
    ii) across all samples selected, each criteria below is satisfied by at least one selected sample among all samples selected.
        - Metrics with >20% variance between Q2 and Q3. Emphasize metrics with exceptionally large percentage changes.
        - Include metrics from the following entities due to past issues:
            --CB Cash Italy
            --CB Correspondent Banking Greece
            --IB Debt Markets Luxembourg
            --CB Trade Finance Brazil
            --PB EMEA UAE
        - Include metrics A1 and C1, which carry higher risk weightings.
        - Include rows where values are zero for both quarters.
        - Include entries from Trade Finance and Correspondent Banking businesses.
        - Include metrics from Cayman Islands, Pakistan, and UAE.
        - Ensure coverage across all Divisions and sub-Divisions.
    
    Return ONLY the new column, "Sample Selected".
"""
    user = f"Audit Data: {sdf}\n\n\nReturn ONLY the column indicating the sampled rows, 'Sample Selected', in Markdown format."

    return {
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ss_resp},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_directory", required=True, help="Path to the directory containing the csv files")
    ap.add_argument("--out_train", default="train.jsonl")
    ap.add_argument("--out_val", default="val.jsonl")
    ap.add_argument("--val_proportion", type=float, default=0.1)
    args = ap.parse_args()

    dframes = find_csv_data(args.root_directory)

    split = int(len(dframes) * args.val_proportion)
    val_recs = dframes[:split]
    train_recs = dframes[split:]

    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(make_example(r), ensure_ascii=False) + "\n")

    with open(args.out_val, "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(make_example(r), ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_recs)} train and {len(val_recs)} val examples.")


if __name__ == "__main__":
    main()
