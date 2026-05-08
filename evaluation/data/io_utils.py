from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def _read_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        # Peek at the first non-whitespace character to decide if the file is JSONL.
        while True:
            ch = f.read(1)
            if not ch:
                return pd.DataFrame()
            if ch.isspace():
                continue
            first_char = ch
            break
        f.seek(0)
        if first_char == "[":
            return pd.read_json(f, orient="records")
        # JSON Lines
        return pd.read_json(f, lines=True)


def load_eval_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load evaluation data from json/jsonl/parquet into a pandas DataFrame.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".json", ".jsonl"}:
        return _read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported evaluation file type: {path.suffix}")
