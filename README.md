# Debiased Skip-gram Experiment

Empirical comparison between standard Skip-gram with Negative Sampling (SGNS) and a novel "Debiased Skip-gram" variant, extending the I-Con framework (Information-Contrastive Learning) to word embeddings.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Download Data (First Run Only)

```bash
python -m debiased_skipgram.data.download
```

This will download and cache:
- Text8 corpus
- SimLex-999
- WordSim-353
- Stanford Rare Words
- Google Analogies

All datasets are cached in `~/.cache/debiased_skipgram/`.

### Run Full Comparison

```bash
python -m debiased_skipgram.experiments.run_comparison --dim 100 --epochs 5 --seeds 42 123 456
```

### Command-Line Options

- `--dim`: Embedding dimension (default: 100)
- `--epochs`: Number of training epochs (default: 5)
- `--seeds`: Random seeds for multiple runs, space-separated (default: 42 123 456)
- `--alphas`: Alpha values for debiased experiments, space-separated (default: 0.2 0.4 0.6)
- `--output-dir`: Output directory for results (default: results)

**Example with custom alpha values:**
```bash
python -m debiased_skipgram.experiments.run_comparison --dim 100 --epochs 5 --alphas 0.01 0.05 0.1 0.2 --output-dir results_small_alphas
```

### Experimental Conditions

1. **Standard SGNS**: Frequency-based negatives, binary targets (baseline)
2. **Uniform Negatives Only**: Uniform negatives, binary targets
3. **Debiased α={alpha}**: Uniform negatives, soft targets (one condition per alpha value specified)

By default, debiased conditions are created for α=0.2, 0.4, and 0.6. You can customize these using the `--alphas` flag. Each condition is run multiple times with different random seeds (specified via `--seeds`) for statistical robustness.

## Results

Results are saved in the `results/` directory:
- `all_results.csv` - All individual runs
- `summary.csv` - Aggregated statistics (mean ± std)
- `comparison.png` - Visualization comparing all conditions
- `embeddings_*.npy` - NumPy embeddings for each condition
- `embeddings_*.vec` - Word2vec text format embeddings

## Project Structure

```
debiased_skipgram/
├── config.py                 # Hyperparameters and configuration
├── data/
│   ├── download.py          # Download Text8 and evaluation datasets
│   ├── corpus.py            # Corpus processing and vocabulary
│   └── evaluation.py        # Load evaluation benchmarks
├── model/
│   ├── skipgram.py          # Core Skip-gram model (PyTorch)
│   ├── negative_sampling.py # Standard and uniform negative sampling
│   └── loss.py              # Standard and debiased loss functions
├── training/
│   ├── trainer.py           # Training loop
│   └── dataloader.py        # Efficient batch generation
├── evaluation/
│   ├── similarity.py        # Word similarity evaluation
│   ├── analogy.py           # Word analogy evaluation
│   └── metrics.py           # Spearman correlation, accuracy
└── experiments/
    └── run_comparison.py    # Main experiment script
```

## Expected Runtime

On M1 MacBook Pro 16GB: **20-40 minutes** for the full comparison.

