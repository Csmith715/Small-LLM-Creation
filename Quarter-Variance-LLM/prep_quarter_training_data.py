import argparse
import pandas as pd
import json
from pathlib import Path


def collect_csv_data(root_dir):
    list_of_dataframes = []
    path = Path(root_dir)
    csv_files = path.glob('**/population.csv')
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df.rename(columns={"Q2_2024": "Q2", "Q3_2024": "Q3"})
        df = df[['Q2', 'Q3']]
        df['Variance'] = [round(((a - b)/b)*100, 0) if b != 0 else 0 for a, b in zip(df['Q2'], df['Q3'])]
        list_of_dataframes.append(df)
    return list_of_dataframes

def make_example(qdf: pd.DataFrame) -> dict:
    qdf1 = qdf[['Q2', 'Q3']].to_markdown(index=False)
    var_resp = qdf['Variance'].to_markdown(index=False)
    sys = ("You are an expert data analyst. Given summarized data from a tabular dataset, you will be asked to perform various statistical analyses. Return ONLY the "
           "quarter-on-quarter variances for each record as a rounded integer.")
    user = f"Quarterly Data: {qdf1}\n\n\nReturn ONLY the quarter-on-quarter variances for each record as a rounded integer."
    assistant = var_resp

    return {
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_directory", required=True, help="Path to the directory containing the csv files")
    ap.add_argument("--out_train", default="train.jsonl")
    ap.add_argument("--out_val", default="val.jsonl")
    ap.add_argument("--val_proportion", type=float, default=0.1)
    args = ap.parse_args()

    dframes = collect_csv_data(args.root_directory)

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
