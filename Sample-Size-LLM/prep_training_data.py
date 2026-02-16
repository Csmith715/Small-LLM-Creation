import argparse
import random
from typing import Dict, List
import json
from pathlib import Path


def combine_audit_jsons(root_dir) -> List[Dict]:
    # combined_data = {}
    combined_data = []
    path = Path(root_dir)

    # Iterate through all .json files in subdirectories
    for json_file in path.glob('**/sample_size_calculation.json'):
        # Get the folder name (e.g., 'ex_00000') to use as a key
        # Commenting out for now, may be useful later to index examples
        # folder_name = json_file.parent.name

        try:
            with open(json_file, 'r') as f:
                combined_data.append(json.load(f))
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return combined_data


def make_example(rec: Dict) -> Dict:
    sys = ("You are an expert data analyst. Given summarized data from a tabular dataset, you will be asked to perform various statistical analyses. Return ONLY the final sample "
           "size as an integer.")

    user = (
        f"Population size: {rec['population_size']}\n"
        f"Confidence level: {rec['confidence_level']}\n"
        f"Tolerable error rate: {rec['tolerable_error']}\n"
        f"Assumed probability of success: {rec['assumed_p']}\n"
        f"Rounding: {rec.get('rounding', 'ceil')}\n"
        f"Used FPC: {rec.get('used_fpc', True)}\n\n"
        "Return ONLY the final sample size."
    )

    assistant = str(rec["final_sample_size"]).strip()

    return {
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_directory", required=True, help="Path to the directory containing sample_size_calculation.json files")
    ap.add_argument("--out_train", default="train.jsonl")
    ap.add_argument("--out_val", default="val.jsonl")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # with open(args.in_json, "r", encoding="utf-8") as f:
    #     records: List[Dict] = json.load(f)
    records = combine_audit_jsons(args.root_directory)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * args.val_frac))
    val_recs = records[:n_val]
    train_recs = records[n_val:]

    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(make_example(r), ensure_ascii=False) + "\n")

    with open(args.out_val, "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(make_example(r), ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_recs)} train and {len(val_recs)} val examples.")


if __name__ == "__main__":
    main()
