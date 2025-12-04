"""Evaluation dataset loaders."""

from .download import (
    download_simlex999,
    download_wordsim353,
    download_rarewords,
    download_google_analogies
)

# Re-export for convenience
__all__ = [
    'download_simlex999',
    'download_wordsim353',
    'download_rarewords',
    'download_google_analogies'
]

