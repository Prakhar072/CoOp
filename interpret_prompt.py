import os
import argparse
from glob import glob
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import re

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip


def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    model = clip.build_model(state_dict or model.state_dict())
    return model


def find_prompt_files(base_dir):
    base_dir = os.path.abspath(base_dir)
    # find all prompt_learner directories and choose the best checkpoint file inside
    prompt_dirs = glob(os.path.join(base_dir, '**', 'prompt_learner'), recursive=True)
    files = {}
    for pd in prompt_dirs:
        run_dir_abs = os.path.dirname(pd)
        rel = os.path.relpath(run_dir_abs, base_dir)
        # collect candidate files
        candidates = [p for p in glob(os.path.join(pd, '*')) if os.path.isfile(p)]
        if not candidates:
            continue

        # prefer files that look like model checkpoints in this order
        def pick_by_substrings(cands, substrs):
            matches = [p for p in cands if any(s in os.path.basename(p) for s in substrs)]
            if matches:
                # prefer most recently modified
                matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return matches[0]
            return None

        selected = None
        # look for .pth.tar-like files
        selected = pick_by_substrings(candidates, ['.pth.tar', '.pth.tar-'])
        if selected is None:
            selected = pick_by_substrings(candidates, ['.tar', 'model.tar'])
        if selected is None:
            selected = pick_by_substrings(candidates, ['.pth'])
        if selected is None:
            selected = pick_by_substrings(candidates, ['.pt'])

        # if there's a 'checkpoint' pointer file, try to read it and resolve
        if selected is None:
            for c in candidates:
                if os.path.basename(c) == 'checkpoint':
                    try:
                        with open(c, 'r') as fh:
                            text = fh.read().strip()
                        # try to extract a plausible target from the file
                        lines = [ln.strip().strip('"').strip('\'') for ln in text.splitlines() if ln.strip()]
                        target = None
                        for ln in reversed(lines):
                            if 'model' in ln or '.pth' in ln or '.tar' in ln:
                                target = ln
                                break
                        if target:
                            if not os.path.isabs(target):
                                candidate = os.path.join(pd, target)
                            else:
                                candidate = target
                            if os.path.exists(candidate):
                                selected = candidate
                                break
                    except Exception:
                        pass

        # fallback: pick the largest file (likely a binary checkpoint)
        if selected is None and candidates:
            try:
                selected = max(candidates, key=lambda p: os.path.getsize(p))
            except Exception:
                selected = candidates[0]

        if selected:
            files[rel] = selected
    return files


def extract_ctx_from_checkpoint(path):
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    sd = None
    if isinstance(loaded, dict):
        sd = loaded.get('state_dict', loaded)
    else:
        raise RuntimeError(f"Unexpected checkpoint format: {path}")

    # direct key
    if 'ctx' in sd:
        return sd['ctx'].float()

    # search for keys that contain ctx
    for k, v in sd.items():
        if k.endswith('.ctx') or k == 'prompt_learner.ctx' or k.endswith('ctx'):
            return v.float()

    raise KeyError("Could not find 'ctx' in checkpoint state dict keys")


def visualize_token_nearest_words(ctx, token_embedding, tokenizer, topk=10):
    # ctx: [L, D] or [C, L, D]
    if ctx.dim() == 2:
        distance = torch.cdist(ctx, token_embedding)
        sorted_idxs = torch.argsort(distance, dim=1)[:, :topk]
        results = []
        for m, idxs in enumerate(sorted_idxs):
            words = [tokenizer.decoder[idx.item()] for idx in idxs]
            dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
            results.append((m, words, dist))
        return results
    elif ctx.dim() == 3:
        C = ctx.shape[0]
        all_results = []
        for c in range(C):
            res = visualize_token_nearest_words(ctx[c], token_embedding, tokenizer, topk=topk)
            all_results.append(res)
        return all_results
    else:
        raise ValueError("Unsupported ctx dim")


def compute_similarity_and_visualize(ctx1, ctx2, out_prefix, plt_enabled=True):
    if out_prefix and os.path.dirname(out_prefix):
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    results = {}
    if ctx1.dim() == 2 and ctx2.dim() == 2:
        avg1 = ctx1.mean(dim=0)
        avg2 = ctx2.mean(dim=0)
        cos_global = F.cosine_similarity(avg1.unsqueeze(0), avg2.unsqueeze(0)).item()
        results['global_cos'] = cos_global
        if ctx1.shape[0] == ctx2.shape[0]:
            per_token = F.cosine_similarity(ctx1, ctx2, dim=1).cpu().numpy()
            results['per_token_cos'] = per_token
        m1 = F.normalize(ctx1, dim=1) @ F.normalize(ctx2, dim=1).t()
        mat = m1.cpu().numpy()
        results['matrix'] = mat
        if plt_enabled and plt:
            try:
                plt.figure(figsize=(6,5))
                plt.imshow(mat, aspect='auto', cmap='viridis')
                plt.colorbar(label='cosine')
                plt.xlabel('method2 tokens')
                plt.ylabel('method1 tokens')
                plt.title(f'Cosine similarity matrix ({mat.shape[0]}x{mat.shape[1]})\nGlobal cos={cos_global:.4f}')
                plt.tight_layout()
                out_path = out_prefix + '_sim.png'
                plt.savefig(out_path)
                print(f"    Saved similarity matrix PNG to {out_path}")
                plt.close()
            except Exception as e:
                print(f"    Warning: Failed to save PNG at {out_prefix}_sim.png: {e}")
    elif ctx1.dim() == 3 and ctx2.dim() == 3 and ctx1.shape[0] == ctx2.shape[0]:
        C = ctx1.shape[0]
        results['per_class'] = []
        for c in range(C):
            resc = compute_similarity_and_visualize(ctx1[c], ctx2[c], f"{out_prefix}_class{c}", plt_enabled=plt_enabled)
            results['per_class'].append(resc)
        return results
    else:
        f1 = ctx1.view(-1)
        f2 = ctx2.view(-1)
        cos = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
        results['global_cos_flatten'] = cos
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to a single learned prompt (keeps original behavior)")
    parser.add_argument("--dir1", help="First output folder (e.g., output_ce)")
    parser.add_argument("--dir2", help="Second output folder (e.g., output_0.2hybrid)")
    parser.add_argument("--topk", type=int, default=10, help="Top-k similar words to show")
    parser.add_argument("--out", default="prompt_analysis_out", help="Directory to save visualizations")
    parser.add_argument("--backbone", default="RN50", help="CLIP backbone name for token embedding")
    args = parser.parse_args()

    tokenizer = SimpleTokenizer()
    clip_model = load_clip_to_cpu(args.backbone)
    token_embedding = clip_model.token_embedding.weight
    print(f"Size of token embedding: {token_embedding.shape}")

    if args.file and not (args.dir1 or args.dir2):
        fpath = args.file
        assert os.path.exists(fpath)
        print(f"Return the top-{args.topk} matched words for {fpath}")
        prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]
        ctx = prompt_learner["ctx"].float()
        print(f"Size of context: {ctx.shape}")
        if ctx.dim() == 2:
            results = visualize_token_nearest_words(ctx, token_embedding, tokenizer, topk=args.topk)
            for m, words, dist in results:
                print(f"{m+1}: {words} {dist}")
        elif ctx.dim() == 3:
            raise NotImplementedError
        return

    if not (args.dir1 and args.dir2):
        parser.error("Either provide --file or both --dir1 and --dir2 for comparison mode")

    dir1 = args.dir1
    dir2 = args.dir2
    files1 = find_prompt_files(dir1)
    files2 = find_prompt_files(dir2)
    common_keys = set(files1.keys()) & set(files2.keys())
    print(f"Found {len(files1)} prompt runs in dir1, {len(files2)} in dir2, {len(common_keys)} matched run-seeds to compare")

    # group by experiment (strip the trailing seed folder)
    group_map = {}
    for rel in common_keys:
        group = os.path.dirname(rel)
        group_map.setdefault(group, []).append(rel)

    print(f"Found {len(group_map)} unique experiments (groups) across seeds")

    os.makedirs(args.out, exist_ok=True)
    summary_rows = []
    commented_lines = []

    def has_shot_tag(s, n):
        # Check for patterns like rn50_16shots, rn50_4shots, 16shot, 4shot, etc.
        return bool(re.search(rf"{n}[-_]?shots?\b", s, flags=re.IGNORECASE))

    groups = sorted(group_map.keys())
    # Build list of groups to process according to shot rules
    groups_to_process = []
    for group in groups:
        dataset = group.split(os.sep)[0].lower()
        if dataset in ('fgvc_aircraft', 'ucf101'):
            # for these datasets only run 4-shot; comment out any 16-shot lines
            if has_shot_tag(group, 16):
                commented_lines.append(f"# {group},excluded: only 4-shot available")
                continue
            if has_shot_tag(group, 4):
                groups_to_process.append(group)
            else:
                commented_lines.append(f"# {group},skipped: no 4-shot variant found")
        else:
            # for all other datasets, only run 16-shot
            if has_shot_tag(group, 16):
                groups_to_process.append(group)
            else:
                commented_lines.append(f"# {group},skipped: not a 16-shot variant")

    print(f"Will process {len(groups_to_process)} groups after shot filtering")

    print("\nGroups found before filtering:")
    for g in groups:
        print(f"  {g}")
    print("\nGroups after filtering:")
    for g in groups_to_process:
        print(f"  {g}")

    for group in groups_to_process:
        seeds = sorted(group_map[group])
        if len(seeds) != 3:
            print(f"Skipping {group}: expected 3 seeds, found {len(seeds)}")
            commented_lines.append(f"# {group},skipped: found {len(seeds)} seeds")
            continue

        seed_vals = []
        seed_matrices = []
        for rel in seeds:
            p1 = files1[rel]
            p2 = files2[rel]
            print(f"\nComparing run: {rel}")
            print(f"  method1: {p1}")
            print(f"  method2: {p2}")
            try:
                ctx1 = extract_ctx_from_checkpoint(p1)
                ctx2 = extract_ctx_from_checkpoint(p2)
            except Exception as e:
                print(f"  Failed to load ctx: {e}")
                seed_vals.append(None)
                continue
            print(f"  ctx shapes: {ctx1.shape} vs {ctx2.shape}")
            # save per-seed PNGs and aggregate later
            out_prefix = os.path.join(args.out, f"{group.replace(os.sep, '_')}_{os.path.basename(rel)}")
            res = compute_similarity_and_visualize(ctx1, ctx2, out_prefix, plt_enabled=True)

            val = None
            if 'global_cos' in res:
                val = res['global_cos']
            elif 'global_cos_flatten' in res:
                val = res['global_cos_flatten']
            elif 'per_class' in res:
                # average per-class values if present
                vals = []
                for rc in res['per_class']:
                    if isinstance(rc, dict):
                        if 'global_cos' in rc:
                            vals.append(rc['global_cos'])
                        elif 'global_cos_flatten' in rc:
                            vals.append(rc['global_cos_flatten'])
                if vals:
                    val = float(np.mean(vals))

            if val is not None:
                seed_vals.append(float(val))

            if 'matrix' in res and isinstance(res['matrix'], (list, tuple, np.ndarray)):
                try:
                    seed_matrices.append(np.array(res['matrix']))
                except Exception:
                    pass

        valid_vals = [v for v in seed_vals if v is not None]
        if len(valid_vals) == 3:
            avg = float(np.mean(valid_vals))
            summary_rows.append((group, avg))
            # save aggregated matrix if matrices exist and align
            if seed_matrices:
                try:
                    shapes = [m.shape for m in seed_matrices]
                    if all(s == shapes[0] for s in shapes):
                        agg = sum(seed_matrices) / len(seed_matrices)
                        if plt is not None:
                            plt.figure(figsize=(6,5))
                            plt.imshow(agg, aspect='auto', cmap='viridis')
                            plt.colorbar(label='cosine')
                            plt.xlabel('method2 tokens')
                            plt.ylabel('method1 tokens')
                            plt.title(f'Avg cosine matrix ({agg.shape[0]}x{agg.shape[1]})\nAvg global cos={avg:.4f}')
                            plt.tight_layout()
                            agg_path = os.path.join(args.out, f"{group.replace(os.sep, '_')}_avg_sim.png")
                            os.makedirs(os.path.dirname(agg_path), exist_ok=True) if os.path.dirname(agg_path) else None
                            plt.savefig(agg_path)
                            print(f"  Saved aggregated similarity matrix to {agg_path}")
                            plt.close()
                        else:
                            print(f"  Warning: matplotlib not available, skipping aggregated PNG for {group}")
                except Exception as e:
                    print(f"  Failed to compute/save aggregated matrix for {group}: {e}")
        else:
            print(f"Skipping {group}: not all seeds succeeded ({len(valid_vals)}/3 valid)")
            commented_lines.append(f"# {group},skipped: only {len(valid_vals)} valid seeds")

    csv_path = os.path.join(args.out, 'summary.csv')
    with open(csv_path, 'w') as fh:
        fh.write('group,avg_global_cos\n')
        for g, a in summary_rows:
            fh.write(f"{g},{a:.6f}\n")
        for c in commented_lines:
            fh.write(c + '\n')

    print(f"\nSaved visualizations and summary to {args.out}")


if __name__ == '__main__':
    main()
