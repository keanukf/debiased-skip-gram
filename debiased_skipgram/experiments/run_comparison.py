"""Main experiment comparing standard SGNS vs Debiased Skip-gram."""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, List

from debiased_skipgram.config import Config
from debiased_skipgram.data.corpus import Corpus
from debiased_skipgram.model.skipgram import SkipGram
from debiased_skipgram.model.negative_sampling import NegativeSampler
from debiased_skipgram.model.loss import SGNSLoss, DebiasedSGNSLoss
from debiased_skipgram.training.dataloader import create_dataloader
from debiased_skipgram.training.trainer import Trainer
from debiased_skipgram.evaluation.similarity import (
    load_simlex999,
    load_wordsim353,
    load_rarewords,
    evaluate_similarity
)
from debiased_skipgram.evaluation.analogy import (
    load_google_analogies,
    evaluate_analogies,
    aggregate_analogy_results
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: Config) -> str:
    """Get available device."""
    if config.device == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif config.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def save_embeddings(
    embeddings: np.ndarray,
    corpus: Corpus,
    output_path: Path,
    format: str = "both"
):
    """
    Save embeddings in .npy and/or .vec format.
    
    Args:
        embeddings: (vocab_size, dim) numpy array
        corpus: Corpus object with vocabulary
        output_path: Base path (without extension)
        format: "npy", "vec", or "both"
    """
    if format in ["npy", "both"]:
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, embeddings)
        print(f"Saved embeddings to {npy_path}")
    
    if format in ["vec", "both"]:
        vec_path = output_path.with_suffix('.vec')
        with open(vec_path, 'w', encoding='utf-8') as f:
            # Write header: vocab_size dim
            f.write(f"{embeddings.shape[0]} {embeddings.shape[1]}\n")
            
            # Write each word and its embedding
            for i, word in enumerate(corpus.idx2word):
                emb_str = ' '.join(f'{val:.6f}' for val in embeddings[i])
                f.write(f"{word} {emb_str}\n")
        
        print(f"Saved embeddings to {vec_path}")


def run_single_experiment(config: Config, seed: int) -> Dict[str, float]:
    """Run a single training + evaluation experiment."""
    print(f"\n{'='*60}")
    print(f"Running experiment with seed {seed}")
    print(f"Config: negative_sampling={config.negative_sampling}, alpha={config.alpha}")
    print(f"{'='*60}\n")
    
    # Set seeds
    set_seed(seed)
    
    # Update device
    config.device = get_device(config)
    
    # Load data
    print("Loading corpus...")
    corpus = Corpus(min_count=config.min_count)
    word_pairs = corpus.get_word_pairs(config.window_size)
    
    print(f"Vocabulary size: {corpus.vocab_size}")
    print(f"Word pairs: {len(word_pairs)}\n")
    
    # Create model
    print("Initializing model...")
    model = SkipGram(corpus.vocab_size, config.embedding_dim).to(config.device)
    
    # Create loss function
    if config.alpha > 0:
        loss_fn = DebiasedSGNSLoss(config.alpha, config.negative_samples)
        print(f"Using DebiasedSGNSLoss with alpha={config.alpha}")
    else:
        loss_fn = SGNSLoss()
        print("Using standard SGNSLoss")
    
    # Create negative sampler
    # Convert word_counts to Counter with indices
    from collections import Counter
    word_counts_by_idx = Counter({
        corpus.word2idx[word]: count 
        for word, count in corpus.word_counts.items()
        if word in corpus.word2idx
    })
    
    negative_sampler = NegativeSampler(
        word_counts_by_idx,
        corpus.vocab_size,
        mode=config.negative_sampling
    )
    print(f"Negative sampling: {config.negative_sampling}\n")
    
    # Create dataloader
    dataloader = create_dataloader(
        word_pairs,
        negative_sampler,
        config.negative_samples,
        config.batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        shuffle=True
    )
    
    # Create trainer
    trainer = Trainer(model, loss_fn, dataloader, config)
    
    # Train
    print("Starting training...")
    history = trainer.train()
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = model.get_embeddings(combine="center")
    
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    
    # Evaluate
    print("\nEvaluating on similarity datasets...")
    results = {}
    
    # SimLex-999
    simlex = load_simlex999()
    rho, found, total = evaluate_similarity(embeddings, corpus.word2idx, simlex)
    results["SimLex_rho"] = rho
    results["SimLex_coverage"] = found / total if total > 0 else 0.0
    print(f"SimLex-999: ρ={rho:.4f}, coverage={found}/{total}")
    
    # WordSim-353
    wordsim = load_wordsim353()
    rho, found, total = evaluate_similarity(embeddings, corpus.word2idx, wordsim)
    results["WordSim_rho"] = rho
    results["WordSim_coverage"] = found / total if total > 0 else 0.0
    print(f"WordSim-353: ρ={rho:.4f}, coverage={found}/{total}")
    
    # RareWords
    rarewords = load_rarewords()
    rho, found, total = evaluate_similarity(embeddings, corpus.word2idx, rarewords)
    results["RareWords_rho"] = rho
    results["RareWords_coverage"] = found / total if total > 0 else 0.0
    print(f"RareWords: ρ={rho:.4f}, coverage={found}/{total}")
    
    # Analogies
    print("\nEvaluating on analogies...")
    analogies = load_google_analogies()
    analogy_results = evaluate_analogies(
        embeddings,
        corpus.word2idx,
        analogies,
        method="3cosadd"
    )
    
    aggregated = aggregate_analogy_results(analogy_results)
    
    if "semantic" in aggregated:
        acc, correct, total = aggregated["semantic"]
        results["analogy_semantic"] = acc
        results["analogy_semantic_correct"] = correct
        results["analogy_semantic_total"] = total
        print(f"Semantic analogies: {acc:.4f} ({correct}/{total})")
    
    if "syntactic" in aggregated:
        acc, correct, total = aggregated["syntactic"]
        results["analogy_syntactic"] = acc
        results["analogy_syntactic_correct"] = correct
        results["analogy_syntactic_total"] = total
        print(f"Syntactic analogies: {acc:.4f} ({correct}/{total})")
    
    if "overall" in aggregated:
        acc, correct, total = aggregated["overall"]
        results["analogy_overall"] = acc
        results["analogy_overall_correct"] = correct
        results["analogy_overall_total"] = total
        print(f"Overall analogies: {acc:.4f} ({correct}/{total})")
    
    # Add training info
    results["final_loss"] = history['epoch_losses'][-1] if history['epoch_losses'] else 0.0
    
    return results, embeddings, corpus


def create_result_plots(df: pd.DataFrame, output_dir: Path):
    """Create bar plots comparing conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics to plot
    metrics = ["SimLex_rho", "RareWords_rho", "analogy_semantic", "analogy_syntactic"]
    metric_labels = {
        "SimLex_rho": "SimLex-999 (ρ)",
        "RareWords_rho": "RareWords (ρ)",
        "analogy_semantic": "Semantic Analogies (Acc)",
        "analogy_syntactic": "Syntactic Analogies (Acc)"
    }
    
    # Dynamically order conditions: Standard SGNS, Uniform Negatives, then Debiased (sorted by alpha)
    all_conditions = df["condition"].unique()
    standard_conditions = [c for c in all_conditions if "Standard SGNS" in c]
    uniform_conditions = [c for c in all_conditions if "Uniform Negatives" in c]
    debiased_conditions = [c for c in all_conditions if "Debiased" in c]
    
    # Sort debiased conditions by alpha value (extract from name)
    def extract_alpha(condition_name):
        if "α=" in condition_name:
            try:
                alpha_str = condition_name.split("α=")[1]
                return float(alpha_str)
            except:
                return 0.0
        return 0.0
    
    debiased_conditions.sort(key=extract_alpha)
    condition_order = standard_conditions + uniform_conditions + debiased_conditions
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, metric in zip(axes, metrics):
        # Aggregate by condition
        summary = df.groupby("condition")[metric].agg(["mean", "std"])
        
        # Reorder
        summary = summary.reindex(condition_order)
        
        # Plot
        x_pos = np.arange(len(summary))
        bars = ax.bar(
            x_pos,
            summary["mean"],
            yerr=summary["std"],
            capsize=5,
            alpha=0.7,
            edgecolor='black'
        )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary.index, rotation=45, ha="right")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_dir / 'comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare standard SGNS vs Debiased Skip-gram"
    )
    parser.add_argument("--dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for multiple runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6],
        help="Alpha values for debiased experiments (default: 0.2 0.4 0.6)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experimental conditions
    conditions = [
        {
            "name": "Standard SGNS",
            "negative_sampling": "frequency",
            "alpha": 0.0
        },
        {
            "name": "Uniform Negatives",
            "negative_sampling": "uniform",
            "alpha": 0.0
        },
    ]
    
    # Add debiased conditions for each alpha value
    for alpha in args.alphas:
        conditions.append({
            "name": f"Debiased α={alpha}",
            "negative_sampling": "uniform",
            "alpha": alpha
        })
    
    all_results = []
    
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Running: {condition['name']}")
        print(f"{'='*60}")
        
        condition_results = []
        
        for seed in args.seeds:
            config = Config(
                embedding_dim=args.dim,
                epochs=args.epochs,
                negative_sampling=condition["negative_sampling"],
                alpha=condition["alpha"],
                seed=seed
            )
            
            results, embeddings, corpus = run_single_experiment(config, seed)
            results["condition"] = condition["name"]
            results["seed"] = seed
            condition_results.append(results)
            
            # Save embeddings for this run
            emb_name = f"embeddings_{condition['name'].replace(' ', '_').replace('α=', 'alpha')}_seed{seed}"
            save_embeddings(
                embeddings,
                corpus,
                output_dir / emb_name,
                format="both"
            )
        
        all_results.extend(condition_results)
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # Save all results
    df.to_csv(output_dir / "all_results.csv", index=False)
    print(f"\nSaved all results to {output_dir / 'all_results.csv'}")
    
    # Aggregate by condition
    summary = df.groupby("condition").agg({
        "SimLex_rho": ["mean", "std"],
        "WordSim_rho": ["mean", "std"],
        "RareWords_rho": ["mean", "std"],
        "analogy_semantic": ["mean", "std"],
        "analogy_syntactic": ["mean", "std"],
        "analogy_overall": ["mean", "std"]
    })
    
    summary.to_csv(output_dir / "summary.csv")
    print(f"Saved summary to {output_dir / 'summary.csv'}")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(summary.to_string())
    
    # Create plots
    create_result_plots(df, output_dir)
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print(f"Results saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

