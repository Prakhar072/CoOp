import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip


# ============================================================
# EXPLICIT PATH PAIRS
# Edit this dict to match your actual checkpoint paths.
# Key = dataset label used in output filenames
# Each entry has:
#   "ce":    path to CE baseline checkpoint
#   "gloss": path to CE+GLoss hybrid checkpoint
#   "shots": label for output ("16shot" or "4shot")
# ============================================================

DATASET_PAIRS = {
    "caltech101": {
        "ce":    "output_ce/caltech101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50",
        "gloss": "output_0.2hybrid/caltech101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "shots": "16shot",
    },
    "dtd": {
        "ce":    "output_ce/dtd/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "gloss": "output_0.2hybrid/dtd/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "shots": "16shot",
    },
    "oxford_flowers": {
        "ce":    "output_ce/oxford_flowers/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "gloss": "output_0.2hybrid/oxford_flowers/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "shots": "16shot",
    },
    "oxford_pets": {
        "ce":    "output_ce/oxford_pets/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50",
        "gloss": "output_0.2hybrid/oxford_pets/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "shots": "16shot",
    },
    "fgvc_aircraft": {
        "ce":    "output_ce/fgvc_aircraft/CoOp/rn50_4shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "gloss": "output_0.2hybrid/fgvc_aircraft/CoOp/rn50_ep50_4shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "shots": "4shot",
    },
    "ucf101": {
        "ce":    "output_ce/ucf101/CoOp/rn50_4shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "gloss": "output_0.2hybrid/ucf101/CoOp/rn50_ep50_4shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
        "shots": "4shot",
    },
}


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


def extract_ctx(path):
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {path}")
    sd = loaded.get("state_dict", loaded)
    if "ctx" in sd:
        return sd["ctx"].float()
    for k, v in sd.items():
        if k.endswith(".ctx") or k == "prompt_learner.ctx":
            return v.float()
    raise KeyError(f"Could not find 'ctx' in checkpoint: {path}\nAvailable keys: {list(sd.keys())}")


def visualize_token_nearest_words(ctx, token_embedding, tokenizer, topk=10):
    """Find closest vocabulary tokens for each context position."""
    distance = torch.cdist(ctx, token_embedding)
    sorted_idxs = torch.argsort(distance, dim=1)[:, :topk]
    results = []
    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dists = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        results.append((m, words, dists))
    return results


def plot_similarity_matrix(ctx_ce, ctx_gloss, dataset, shots, out_dir):
    """
    Plot cosine similarity matrix between CE and GLoss prompt tokens.
    Rows = CE tokens, Cols = GLoss tokens.
    """
    mat = (F.normalize(ctx_ce, dim=1) @ F.normalize(ctx_gloss, dim=1).t()).cpu().numpy()
    global_cos = F.cosine_similarity(
        ctx_ce.mean(dim=0, keepdim=True),
        ctx_gloss.mean(dim=0, keepdim=True)
    ).item()
    per_token_cos = F.cosine_similarity(ctx_ce, ctx_gloss, dim=1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: similarity matrix
    im = axes[0].imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0], label='Cosine Similarity')
    axes[0].set_xlabel('CE+GLoss tokens')
    axes[0].set_ylabel('CE tokens')
    axes[0].set_title(f'{dataset} ({shots})\nToken Similarity Matrix\nGlobal cos={global_cos:.4f}')
    axes[0].set_xticks(range(ctx_gloss.shape[0]))
    axes[0].set_yticks(range(ctx_ce.shape[0]))

    # Right: per-token cosine similarity
    axes[1].bar(range(len(per_token_cos)), per_token_cos, color='steelblue', alpha=0.8)
    axes[1].axhline(y=global_cos, color='red', linestyle='--', label=f'Global mean={global_cos:.4f}')
    axes[1].set_xlabel('Context Token Position')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title(f'{dataset} ({shots})\nPer-token Similarity: CE vs CE+GLoss')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].set_xticks(range(len(per_token_cos)))

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_{shots}_similarity.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    return global_cos, per_token_cos, mat


def plot_prompt_heatmaps(ctx_ce, ctx_gloss, dataset, shots, out_dir):
    """
    Side-by-side heatmaps of the raw prompt vectors.
    """
    vmin = min(ctx_ce.min().item(), ctx_gloss.min().item())
    vmax = max(ctx_ce.max().item(), ctx_gloss.max().item())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im1 = axes[0].imshow(ctx_ce.cpu().numpy(), aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_title(f'CE Baseline\n{dataset} ({shots})')
    axes[0].set_xlabel('Embedding Dimension (512)')
    axes[0].set_ylabel('Context Token Position')

    im2 = axes[1].imshow(ctx_gloss.cpu().numpy(), aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_title(f'CE+GLoss Hybrid\n{dataset} ({shots})')
    axes[1].set_xlabel('Embedding Dimension (512)')
    axes[1].set_ylabel('Context Token Position')

    plt.suptitle(f'Learned Prompt Vectors: CE vs CE+GLoss\n{dataset} ({shots})', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_{shots}_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_norm_comparison(ctx_ce, ctx_gloss, dataset, shots, out_dir):
    """
    Compare L2 norms of prompt tokens between CE and CE+GLoss.
    """
    norm_ce = ctx_ce.norm(dim=1).cpu().numpy()
    norm_gloss = ctx_gloss.norm(dim=1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(norm_ce))
    ax.plot(x, norm_ce, marker='o', label='CE Baseline', linewidth=2)
    ax.plot(x, norm_gloss, marker='s', label='CE+GLoss', linewidth=2)
    ax.set_xlabel('Context Token Position')
    ax.set_ylabel('L2 Norm')
    ax.set_title(f'Prompt Token Magnitudes: CE vs CE+GLoss\n{dataset} ({shots})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_{shots}_norms.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_summary_global_cos(summary, out_dir):
    """
    Bar chart of global cosine similarity across all datasets.
    Lower = more different prompts between CE and CE+GLoss.
    """
    datasets = list(summary.keys())
    cos_vals = [summary[d]['global_cos'] for d in datasets]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(datasets, cos_vals, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect similarity')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Global Cosine Similarity')
    ax.set_title('Prompt Similarity: CE Baseline vs CE+GLoss\n(Lower = More Different Prompts Learned)')
    ax.set_ylim([0, 1.05])
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, cos_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    out_path = os.path.join(out_dir, "summary_global_cos.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary chart: {out_path}")


def print_nearest_words(ctx_ce, ctx_gloss, dataset, shots, tokenizer, token_embedding, topk):
    """Print nearest vocabulary words for both CE and CE+GLoss prompts."""
    print(f"\n  [{dataset} {shots}] Nearest words in CE baseline:")
    results_ce = visualize_token_nearest_words(ctx_ce, token_embedding, tokenizer, topk=topk)
    for pos, words, dists in results_ce:
        print(f"    Token {pos+1:2d}: {words[:5]}")

    print(f"\n  [{dataset} {shots}] Nearest words in CE+GLoss hybrid:")
    results_gloss = visualize_token_nearest_words(ctx_gloss, token_embedding, tokenizer, topk=topk)
    for pos, words, dists in results_gloss:
        print(f"    Token {pos+1:2d}: {words[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="prompt_analysis_out", help="Directory to save visualizations")
    parser.add_argument("--topk", type=int, default=5, help="Top-k nearest words to print")
    parser.add_argument("--backbone", default="RN50", help="CLIP backbone name")
    parser.add_argument("--no-words", action="store_true", help="Skip nearest word analysis")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading CLIP tokenizer and token embeddings...")
    tokenizer = SimpleTokenizer()
    clip_model = load_clip_to_cpu(args.backbone)
    token_embedding = clip_model.token_embedding.weight.detach()
    print(f"Token embedding size: {token_embedding.shape}")

    summary = {}

    for dataset, paths in DATASET_PAIRS.items():
        shots = paths["shots"]
        path_ce = paths["ce"]
        path_gloss = paths["gloss"]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({shots})")
        print(f"  CE path:    {path_ce}")
        print(f"  GLoss path: {path_gloss}")

        # Check paths exist
        if not os.path.exists(path_ce):
            print(f"  ⚠ SKIPPING: CE checkpoint not found: {path_ce}")
            continue
        if not os.path.exists(path_gloss):
            print(f"  ⚠ SKIPPING: GLoss checkpoint not found: {path_gloss}")
            continue

        # Load ctx vectors
        try:
            ctx_ce = extract_ctx(path_ce)
            ctx_gloss = extract_ctx(path_gloss)
        except Exception as e:
            print(f"  ⚠ SKIPPING: Failed to load ctx: {e}")
            continue

        print(f"  ctx shape CE: {ctx_ce.shape}, GLoss: {ctx_gloss.shape}")

        # Generate all plots
        global_cos, per_token_cos, mat = plot_similarity_matrix(ctx_ce, ctx_gloss, dataset, shots, args.out)
        plot_prompt_heatmaps(ctx_ce, ctx_gloss, dataset, shots, args.out)
        plot_norm_comparison(ctx_ce, ctx_gloss, dataset, shots, args.out)

        # Print nearest words
        if not args.no_words:
            print_nearest_words(ctx_ce, ctx_gloss, dataset, shots, tokenizer, token_embedding, args.topk)

        # Store summary
        summary[dataset] = {
            "global_cos": global_cos,
            "per_token_mean": float(np.mean(per_token_cos)),
            "per_token_min": float(np.min(per_token_cos)),
            "shots": shots,
        }

        print(f"  Global cosine similarity: {global_cos:.4f}")
        print(f"  Per-token mean similarity: {np.mean(per_token_cos):.4f}")
        print(f"  Most diverged token: position {np.argmin(per_token_cos)} (cos={np.min(per_token_cos):.4f})")

    # Summary chart across all datasets
    if summary:
        plot_summary_global_cos(summary, args.out)

    # Save CSV summary
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,shots,global_cos,per_token_mean,per_token_min\n")
        for d, v in summary.items():
            f.write(f"{d},{v['shots']},{v['global_cos']:.6f},{v['per_token_mean']:.6f},{v['per_token_min']:.6f}\n")
    print(f"\nSaved summary CSV: {csv_path}")
    print(f"\nDone! All outputs saved to: {args.out}/")


if __name__ == "__main__":
    main()