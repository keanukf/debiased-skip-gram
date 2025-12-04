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

### Experimental Conditions

1. **Standard SGNS**: Frequency-based negatives, binary targets (baseline)
2. **Uniform Negatives Only**: Uniform negatives, binary targets
3. **Debiased α=0.2**: Uniform negatives, soft targets
4. **Debiased α=0.4**: Uniform negatives, soft targets
5. **Debiased α=0.6**: Uniform negatives, soft targets

Each condition is run 3 times with different random seeds for statistical robustness.

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

