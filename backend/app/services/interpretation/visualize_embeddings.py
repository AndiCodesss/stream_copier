"""t-SNE visualization of frozen transformer embeddings colored by label.

Usage:
    python -m app.services.interpretation.visualize_embeddings \
        --model distilbert-base-uncased \
        --output plots/tsne_distilbert.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE

from app.services.interpretation.benchmark_models import (
    LABELS,
    _LABEL_TO_ID,
    load_examples_grouped,
    _DEFAULT_DATASET,
)
from app.services.interpretation.train_local_classifier import (
    _embed_prompts,
    _resolve_device,
)


_LABEL_NAMES = [label.value for label in LABELS]

_COLORS = [
    "#999999",  # NO_ACTION (grey)
    "#2ecc71",  # ENTER_LONG (green)
    "#e74c3c",  # ENTER_SHORT (red)
    "#f39c12",  # TRIM (orange)
    "#9b59b6",  # EXIT_ALL (purple)
    "#3498db",  # MOVE_STOP (blue)
    "#1abc9c",  # MOVE_TO_BREAKEVEN (teal)
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("plots/tsne_embeddings.pdf"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Load all examples (no rebalancing — we want the true distribution)
    examples, _ = load_examples_grouped(dataset=args.dataset)
    texts = [ex["prompt"] for ex in examples]
    labels = np.array([_LABEL_TO_ID[ex["label"]] for ex in examples])
    print(f"Loaded {len(texts)} examples")

    # Encode
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    device = _resolve_device(torch, args.device)
    model_path = snapshot_download(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()

    embeddings = _embed_prompts(
        texts=texts,
        tokenizer=tokenizer,
        model=encoder,
        torch_module=torch,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    ).numpy()
    print(f"Embeddings shape: {embeddings.shape}")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)
    print(f"t-SNE done, KL divergence: {tsne.kl_divergence_:.4f}")

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot NO_ACTION first (background), then action classes on top
    for label_id in range(len(LABELS)):
        mask = labels == label_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=_COLORS[label_id],
            label=f"{_LABEL_NAMES[label_id]} ({mask.sum()})",
            s=8 if label_id == 0 else 16,
            alpha=0.3 if label_id == 0 else 0.7,
            edgecolors="none",
        )

    ax.set_title(f"t-SNE of {args.model} embeddings ({len(texts)} examples)")
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
