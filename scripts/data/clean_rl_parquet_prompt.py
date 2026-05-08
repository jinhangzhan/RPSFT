#!/usr/bin/env python3
"""Drop RL parquet rows whose prompt field is malformed."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd


def normalize_prompt(prompt: Any):
    if prompt is None:
        return None
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    elif isinstance(prompt, tuple):
        prompt = list(prompt)
    if not isinstance(prompt, list) or len(prompt) == 0:
        return None

    normalized = []
    for msg in prompt:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            return None
        normalized.append({"role": str(role), "content": str(content)})
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compression", default="none", choices=["none", "snappy", "gzip", "brotli", "zstd", "lz4"])
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if "prompt" not in df.columns:
        raise ValueError(f"Missing prompt column. Found: {list(df.columns)}")

    prompts = [normalize_prompt(p) for p in df["prompt"].tolist()]
    valid_mask = [p is not None for p in prompts]
    valid_df = df.loc[valid_mask].copy()
    valid_df["prompt"] = [p for p in prompts if p is not None]

    compression = None if args.compression == "none" else args.compression
    valid_df.to_parquet(args.output, compression=compression, index=False)
    print(f"input_rows={len(df)} output_rows={len(valid_df)} removed={len(df)-len(valid_df)} output={args.output}")


if __name__ == "__main__":
    main()
