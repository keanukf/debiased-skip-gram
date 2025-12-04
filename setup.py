"""Setup script for debiased-skip-gram package."""

from setuptools import setup, find_packages

setup(
    name="debiased-skip-gram",
    version="0.1.0",
    description="Empirical comparison of standard and debiased Skip-gram word embeddings",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
    ],
    python_requires=">=3.8",
)

