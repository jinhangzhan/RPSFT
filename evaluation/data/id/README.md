# ID Evaluation Datasets

JSONL files in this directory contain in-distribution evaluation sets (e.g., AIME, AMC, Math500).

Use `python evaluation/data/convert_to_json.py` to regenerate them from the original parquet
sources under `evaluation/data/`. Each record is stored as a single JSON object per line with the
columns required by `evaluation/eval_local.py`.
