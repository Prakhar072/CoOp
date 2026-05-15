#!/usr/bin/env python3
"""
Batch-extract nearest token words for multiple prompt-learner checkpoints.

Create a text file with one checkpoint path per line (absolute or repository-relative),
then run this script to produce a CSV with the top-k token words for each model.

Example:
  python scripts/extract_prompt_tokens.py --models models.txt --out tokens.csv --topk 5

The script re-uses helper functions from interpret_prompt.py and requires the same
environment (PyTorch, CLIP code available in the repo).
"""

import argparse
import csv
import os
from pathlib import Path

import torch

import interpret_prompt


def load_clip_and_tokenizer(backbone="RN50"):
    tokenizer = interpret_prompt.SimpleTokenizer()
    clip_model = interpret_prompt.load_clip_to_cpu(backbone)
    token_embedding = clip_model.token_embedding.weight
    return tokenizer, token_embedding


def process_checkpoint(path, tokenizer, token_embedding, topk):
    ctx = interpret_prompt.extract_ctx_from_checkpoint(path)
    entries = []
    if ctx.dim() == 2:
        nearest = interpret_prompt.visualize_token_nearest_words(ctx, token_embedding, tokenizer, topk=topk)
        for m, words, dists in nearest:
            try:
                dists_f = [float(x) for x in dists]
            except Exception:
                dists_f = dists
            entries.append({'class': '', 'token_idx': int(m), 'words': words, 'dists': dists_f})
    elif ctx.dim() == 3:
        C = ctx.shape[0]
        for c in range(C):
            nearest = interpret_prompt.visualize_token_nearest_words(ctx[c], token_embedding, tokenizer, topk=topk)
            for m, words, dists in nearest:
                try:
                    dists_f = [float(x) for x in dists]
                except Exception:
                    dists_f = dists
                entries.append({'class': int(c), 'token_idx': int(m), 'words': words, 'dists': dists_f})
    else:
        raise ValueError(f"Unsupported ctx dim: {ctx.dim()}")
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="Text file with one checkpoint path per line")
    parser.add_argument("--out", required=True, help="Output CSV file path")
    parser.add_argument("--topk", type=int, default=10, help="Top-k words per token")
    parser.add_argument("--backbone", default="RN50", help="CLIP backbone name")
    args = parser.parse_args()

    tokenizer, token_embedding = load_clip_and_tokenizer(args.backbone)

    with open(args.models, 'r') as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith('#')]

    rows = []
    for model_path in lines:
        if not os.path.exists(model_path):
            print(f"Warning: path not found, skipping: {model_path}")
            continue
        print(f"Processing: {model_path}")
        try:
            entries = process_checkpoint(model_path, tokenizer, token_embedding, args.topk)
        except Exception as e:
            print(f"  Failed to load/process {model_path}: {e}")
            continue
        for ent in entries:
            row = {'model': model_path, 'class': ent['class'], 'token_idx': ent['token_idx']}
            for i in range(args.topk):
                wcol = f'word_{i+1}'
                dcol = f'dist_{i+1}'
                try:
                    row[wcol] = ent['words'][i]
                except Exception:
                    row[wcol] = ''
                try:
                    val = ent['dists'][i]
                    row[dcol] = f"{float(val):.6f}" if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.','',1).isdigit()) else str(val)
                except Exception:
                    row[dcol] = ''
            rows.append(row)

    # write CSV
    fieldnames = ['model', 'class', 'token_idx']
    for i in range(args.topk):
        fieldnames += [f'word_{i+1}', f'dist_{i+1}']

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as csvf:
        import csv as _csv
        writer = _csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == '__main__':
    main()
