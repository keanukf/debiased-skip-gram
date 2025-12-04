"""Download and cache datasets for the experiment."""

import os
import zipfile
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm


CACHE_DIR = Path.home() / ".cache" / "debiased_skipgram"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: Path, desc: Optional[str] = None) -> Path:
    """Download a file with progress bar."""
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return dest_path
    
    print(f"Downloading {desc or url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc or "Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded to {dest_path}")
    return dest_path


def download_text8() -> Path:
    """Download Text8 corpus."""
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = CACHE_DIR / "text8.zip"
    text_path = CACHE_DIR / "text8.txt"
    
    if text_path.exists():
        print(f"Text8 already exists: {text_path}")
        return text_path
    
    # Download zip
    download_file(url, zip_path, "Text8 corpus")
    
    # Extract
    print("Extracting Text8...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(CACHE_DIR)
    
    # Rename if needed
    extracted = CACHE_DIR / "text8"
    if extracted.exists() and not text_path.exists():
        extracted.rename(text_path)
    
    return text_path


def download_simlex999() -> Path:
    """Download SimLex-999 dataset."""
    # SimLex-999 is available as a direct download
    url = "https://fh295.github.io/SimLex-999.txt"
    dest_path = CACHE_DIR / "SimLex-999.txt"
    
    if dest_path.exists():
        print(f"SimLex-999 already exists: {dest_path}")
        return dest_path
    
    download_file(url, dest_path, "SimLex-999")
    return dest_path


def download_wordsim353() -> Path:
    """Download WordSim-353 dataset."""
    url = "http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip"
    zip_path = CACHE_DIR / "wordsim353.zip"
    dest_dir = CACHE_DIR / "wordsim353"
    dest_path = dest_dir / "combined.csv"
    
    if dest_path.exists():
        print(f"WordSim-353 already exists: {dest_path}")
        return dest_path
    
    # Download zip
    download_file(url, zip_path, "WordSim-353")
    
    # Extract
    print("Extracting WordSim-353...")
    dest_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    
    return dest_path


def download_rarewords() -> Path:
    """Download Stanford Rare Words dataset."""
    # Rare Words dataset URL
    url = "https://nlp.stanford.edu/~lmthang/morphoNLM/wordvecs/rarewords.txt"
    dest_path = CACHE_DIR / "rarewords.txt"
    
    if dest_path.exists():
        print(f"Rare Words already exists: {dest_path}")
        return dest_path
    
    download_file(url, dest_path, "Stanford Rare Words")
    return dest_path


def download_google_analogies() -> Path:
    """Download Google analogies dataset (questions-words.txt)."""
    # Google analogies dataset from word2vec
    url = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt"
    dest_path = CACHE_DIR / "questions-words.txt"
    
    if dest_path.exists():
        print(f"Google analogies already exists: {dest_path}")
        return dest_path
    
    download_file(url, dest_path, "Google analogies")
    return dest_path


def download_all():
    """Download all required datasets."""
    print("Downloading all datasets...")
    print(f"Cache directory: {CACHE_DIR}\n")
    
    download_text8()
    download_simlex999()
    download_wordsim353()
    download_rarewords()
    download_google_analogies()
    
    print("\nAll datasets downloaded successfully!")


if __name__ == "__main__":
    download_all()

