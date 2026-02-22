# Small LLM Creation

The purpose of this project is to convert small tasks from agentic problems into viable small language models capable of
performing at the same level of efficacy.

## Setup

```bash
uv sync
```
To include Jupyter notebooks and development tools:
```bash
uv sync --extras dev
```

## General Fine-Tuning Actions

### Fine-tune on a small dataset

For the Sample Size LLM example:
```bash
python src/fine_tune_model.py \
--model_id Qwen/Qwen2.5-1.5B-Instruct \
--train_jsonl src/models/Sample-Size-LLM/train.jsonl \
--val_jsonl src/models/Sample-Size-LLM/val.jsonl
```
This will fine-tune the Qwen2.5 model on a previously created dataset. It assumes a max sequence length of 256 tokens. 
However, this parameter can be modified with a `--max_seq_len` flag.

### Carry out inference on the model

```python
from src.fine_tune_inference import FineTuneInference

message = [
    {
        'role': 'system',
        'content': 'You are an expert data analyst. Given summarized data from a tabular dataset, you will be asked to perform various statistical analyses. Return ONLY the final sample size as an integer.'},
    {
        'role': 'user',
        'content': 'Population size: 167\nConfidence level: 0.9\nTolerable error rate: 0.1\nAssumed probability of success: 0.5\nRounding: ceil\nUsed FPC: True\n\nReturn ONLY the final sample size.'
    }
]

inference = FineTuneInference(adapter_dir='src/models/Sample-Size-LLM/sample-size-sft-lora')
inference.generate(message)
```
In most situations, both the fine-tuning and inference steps can be applied to any dataset.

## Sample Size LLM Specific Actions

### Create a dataset

This script will create a directory of 400 synthetic data examples. Each example dataset will consist of a sythesized 
csv file `population.csv` and a summary JSON file `sample_size_calculation.json` that statistical inferences from the csv file.

```bash
python src/models/Sample-Size-LLM/make_audit_sample_size_synth.py --n_examples 400 --write_sft_jsonl
```

An example of the output JSON file is shown below:
```json
{
  "confidence_level": 0.9,
  "z_value": 1.64485,
  "tolerable_error": 0.1,
  "assumed_p": 0.5,
  "n0_unbounded": 67.63828806249998,
  "population_size": 203,
  "finite_population_corrected_n": 50.92219126352285,
  "final_sample_size": 51,
  "used_fpc": true,
  "rounding": "ceil",
  "formula_notes": {
    "n0": "(Z^2 * p * (1-p)) / E^2",
    "fpc": "(N * n0) / (N + n0 - 1)"
  }
}
```

### Generate Train and Test Datasets

This script will create a train and test dataset from the synthetic data examples found in the `audit_synth` 
directory created in the previous step. It also formats the data into an acceptable messaging format for fine-tuning.

```bash
python src/models/Sample-Size-LLM/prep_training_data.py \
 --root_directory "data/audit_synth" \
 --out_train src/models/Sample-Size-LLM/train.jsonl \
 --out_val src/models/Sample-Size-LLM/val.jsonl
```
Each training example will have the following structure:
```json
{
  "messages": [
    {
      "role": "system", 
      "content": "You are an expert data analyst. Given summarized data from a tabular dataset, you will be asked to perform various statistical analyses. Return ONLY the final sample size as an integer."
    }, 
    {
      "role": "user", 
      "content": "Population size: 75\nConfidence level: 0.9\nTolerable error rate: 0.1\nAssumed probability of success: 0.5\nRounding: ceil\nUsed FPC: True\n\nReturn ONLY the final sample size."
    }, 
    {
      "role": "assistant", 
      "content": "36"
    }
  ]
}
```


















