# Debiased Skip-gram: Extending Information-Contrastive Learning to Word Embeddings

## Abstract

This project presents an empirical investigation into debiasing word embeddings through the application of Information-Contrastive Learning (I-Con) principles to Skip-gram with Negative Sampling (SGNS). Standard SGNS employs binary classification targets (positive pairs = 1, negative pairs = 0) and frequency-based negative sampling, which can introduce distributional biases that propagate into the learned representations. We propose a debiased variant that combines uniform negative sampling with soft target labels derived from I-Con theory, systematically evaluating its impact on downstream semantic and syntactic tasks.

## Theoretical Framework

### Information-Contrastive Learning (I-Con)

The I-Con framework addresses bias in contrastive learning by recognizing that hard binary labels (0/1) may not accurately reflect the true information-theoretic relationships between samples. Instead, I-Con proposes using **soft targets** that account for the inherent uncertainty and information content in the contrastive setup.

In the context of word embeddings, the standard SGNS objective treats all negative samples as equally "negative" (target = 0), despite the fact that:

1. Some negatives may be semantically related to the positive context
2. The information content varies across different negative samples
3. Frequency-based negative sampling creates systematic biases toward common words

### Debiased Skip-gram Formulation

Our debiased variant modifies the standard SGNS loss function in two key ways:

**1. Uniform Negative Sampling**: Replace frequency-based sampling $P_n(j) \propto f(j)^{0.75}$ with uniform sampling $P_n(j) = \frac{1}{V}$ to eliminate frequency bias.

**2. Soft Target Labels**: Replace binary targets with information-theoretic soft targets:

- **Positive target**: $t_{\text{pos}} = 1 - \alpha \cdot \frac{k}{k+1}$
- **Negative target**: $t_{\text{neg}} = \frac{\alpha}{k+1}$

where $k$ is the number of negative samples and $\alpha \in [0,1]$ is a debiasing parameter controlling the strength of the soft target regularization.

The loss function becomes:

$$L = -\left[t_{\text{pos}} \cdot \log(\sigma(s_{\text{pos}})) + (1-t_{\text{pos}}) \cdot \log(\sigma(-s_{\text{pos}}))\right] - \sum_i \left[t_{\text{neg}} \cdot \log(\sigma(s_{\text{neg}_i})) + (1-t_{\text{neg}}) \cdot \log(\sigma(-s_{\text{neg}_i}))\right]$$

As $\alpha \to 0$, the model recovers standard binary targets. As $\alpha$ increases, the soft targets provide a smoother learning signal that accounts for the information-theoretic structure of the contrastive space.

## Research Questions

This work addresses several fundamental questions:

1. **Does uniform negative sampling alone improve embedding quality?** By isolating the effect of negative sampling distribution, we can assess whether frequency bias is a primary source of suboptimal representations.

2. **Do soft targets derived from I-Con theory improve upon binary targets?** The soft target formulation encodes the hypothesis that not all negatives are equally informative, and that the learning signal should reflect this structure.

3. **What is the optimal debiasing strength α?** Different values of α trade off between hard binary learning and fully soft information-theoretic learning, with implications for both representation quality and training dynamics.

4. **How do these modifications affect different aspects of word representations?** We evaluate on semantic similarity, rare word representations, and both semantic and syntactic analogies to understand the differential impact across linguistic phenomena.

## Experimental Design

### Baseline Conditions

1. **Standard SGNS** (baseline): Frequency-based negative sampling $P_n(j) \propto f(j)^{0.75}$ with binary targets ($t_{\text{pos}}=1$, $t_{\text{neg}}=0$). This replicates the standard word2vec setup.

2. **Uniform Negatives**: Uniform negative sampling $P_n(j) = \frac{1}{V}$ with binary targets. This isolates the effect of negative sampling distribution.

3. **Debiased α={α}**: Uniform negative sampling with soft targets parameterized by α. Multiple α values are tested to explore the debiasing parameter space.

### Evaluation Protocol

All models are evaluated on established benchmarks:

- **Semantic Similarity**: SimLex-999 (focus on true semantic relatedness), WordSim-353 (general similarity), Stanford Rare Words (low-frequency word quality)
- **Word Analogies**: Google Analogy Test Set (semantic and syntactic categories)

Each experimental condition is run with multiple random seeds (default: 42, 123, 456) to ensure statistical robustness. Results are reported as mean ± standard deviation across runs.

### Implementation Details

- **Corpus**: Text8 (100M tokens, preprocessed Wikipedia text)
- **Vocabulary**: Minimum word frequency = 5
- **Model**: Skip-gram with separate center and context embeddings
- **Training**: Adam optimizer, learning rate 0.025 with decay, window size 5, 5 negative samples per positive pair
- **Embeddings**: L2-normalized center embeddings for evaluation

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

## Results

Results are saved in the `results/` directory:

- `all_results.csv` - All individual runs with full metrics
- `summary.csv` - Aggregated statistics (mean ± std) by condition
- `comparison.png` - Visualization comparing all conditions across key metrics
- `embeddings_*.npy` - NumPy embeddings for each condition and seed
- `embeddings_*.vec` - Word2vec text format embeddings for compatibility

The summary includes:

- Spearman correlation ($\rho$) on similarity datasets
- Accuracy on semantic and syntactic analogies
- Coverage statistics (vocabulary overlap with evaluation sets)
- Training loss trajectories

## Project Structure

```text
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
│   ├── trainer.py           # Training loop with logging
│   └── dataloader.py        # Efficient batch generation
├── evaluation/
│   ├── similarity.py        # Word similarity evaluation
│   ├── analogy.py           # Word analogy evaluation
│   └── metrics.py           # Spearman correlation, accuracy
└── experiments/
    └── run_comparison.py    # Main experiment script
```

## Key Contributions

This work demonstrates:

1. **Theoretical Extension**: First application of I-Con soft targets to word embedding learning, providing a principled framework for debiasing contrastive objectives.

2. **Systematic Ablation**: Controlled comparison isolating the effects of negative sampling distribution and target label formulation.

3. **Empirical Analysis**: Comprehensive evaluation across multiple linguistic tasks to understand differential impacts on semantic vs. syntactic representations.

4. **Reproducible Implementation**: Clean, modular codebase enabling further research into debiasing techniques for distributional semantics.
