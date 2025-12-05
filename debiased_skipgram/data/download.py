"""Download and cache datasets for the experiment."""

import os
import zipfile
import requests
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


CACHE_DIR = Path.home() / ".cache" / "debiased_skipgram"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: Path, desc: Optional[str] = None, timeout: int = 30) -> Path:
    """Download a file with progress bar."""
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return dest_path
    
    print(f"Downloading {desc or url}...")
    response = requests.get(url, stream=True, timeout=timeout)
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
    url = "https://fh295.github.io/SimLex-999.zip"
    zip_path = CACHE_DIR / "SimLex-999.zip"
    dest_path = CACHE_DIR / "SimLex-999.txt"
    
    if dest_path.exists():
        print(f"SimLex-999 already exists: {dest_path}")
        # Check if it's a valid data file (has header row)
        try:
            with open(dest_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('word1') and '\t' in first_line:
                    return dest_path
        except Exception:
            pass
    
    download_file(url, zip_path, "SimLex-999")
    
    # Extract and find the actual file
    print("Extracting SimLex-999...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()
        # Extract all files
        zip_ref.extractall(CACHE_DIR)
        
        # Find the extracted .txt file recursively (could be in a subdirectory)
        extracted_file = None
        for filename in file_list:
            if filename.endswith('.txt') and 'SimLex' in filename:
                # Try both direct path and path with subdirectory
                extracted_path = CACHE_DIR / filename
                if extracted_path.exists():
                    # Check if it's the actual data file (has header row)
                    try:
                        with open(extracted_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline().strip()
                            if first_line.startswith('word1') and '\t' in first_line:
                                extracted_file = extracted_path
                                break
                    except Exception:
                        pass
        
        # If not found in file_list paths, search recursively
        if extracted_file is None:
            for txt_file in CACHE_DIR.rglob("SimLex*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line.startswith('word1') and '\t' in first_line:
                            extracted_file = txt_file
                            break
                except Exception:
                    continue
        
        # If we found a file and it's not already the dest_path, copy it
        if extracted_file and extracted_file != dest_path:
            shutil.copy2(extracted_file, dest_path)
    
    if not dest_path.exists():
        raise FileNotFoundError(f"Could not find SimLex-999.txt after extraction. Files in zip: {file_list}")
    
    return dest_path


def download_wordsim353() -> Path:
    """Download WordSim-353 dataset."""
    dest_dir = CACHE_DIR / "wordsim353"
    dest_path = dest_dir / "combined.csv"
    
    if dest_path.exists():
        print(f"WordSim-353 already exists: {dest_path}")
        return dest_path
    
    dest_dir.mkdir(exist_ok=True)
    
    # First try Kaggle (most reliable)
    if KAGGLE_AVAILABLE:
        try:
            print("Downloading WordSim-353 from Kaggle...")
            kaggle_path = kagglehub.dataset_download("julianschelb/wordsim353-crowd")
            print(f"Downloaded from Kaggle to: {kaggle_path}")
            
            # Find combined.csv in the Kaggle download directory
            kaggle_path_obj = Path(kaggle_path)
            combined_csv = None
            
            # Search for combined.csv in the downloaded directory
            for file_path in kaggle_path_obj.rglob("combined.csv"):
                combined_csv = file_path
                break
            
            # If not found, search for any CSV file that might be the dataset
            if combined_csv is None:
                csv_files = list(kaggle_path_obj.rglob("*.csv"))
                if csv_files:
                    # Try to find the right one - might be wordsim353.csv or similar
                    for csv_file in csv_files:
                        if "wordsim" in csv_file.name.lower() or "353" in csv_file.name:
                            combined_csv = csv_file
                            break
                    # If still not found, use the first CSV
                    if combined_csv is None:
                        combined_csv = csv_files[0]
            
            if combined_csv and combined_csv.exists():
                # Copy to our destination
                shutil.copy2(combined_csv, dest_path)
                print(f"Copied WordSim-353 to {dest_path}")
                return dest_path
            else:
                print(f"Warning: Could not find combined.csv in Kaggle download. Files found: {list(kaggle_path_obj.rglob('*'))}")
        except Exception as e:
            print(f"Failed to download from Kaggle: {e}")
            print("Falling back to alternative sources...")
    
    # Fallback to direct URLs if Kaggle fails or is not available
    zip_urls = [
        "https://github.com/PrincetonML/SIF/raw/master/data/wordsim353.zip",
        "https://raw.githubusercontent.com/PrincetonML/SIF/master/data/wordsim353.zip",
        "http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip",
        "https://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip",
    ]
    csv_urls = [
        "https://raw.githubusercontent.com/PrincetonML/SIF/master/data/wordsim353/combined.csv",
        "https://github.com/PrincetonML/SIF/raw/master/data/wordsim353/combined.csv",
    ]
    
    zip_path = CACHE_DIR / "wordsim353.zip"
    last_error = None
    zip_downloaded = False
    
    # Try downloading zip files
    for url in zip_urls:
        try:
            print(f"Trying to download WordSim-353 from {url}...")
            download_file(url, zip_path, "WordSim-353")
            zip_downloaded = True
            break
        except Exception as e:
            last_error = e
            if zip_path.exists():
                zip_path.unlink()  # Remove failed download
            continue
    
    if zip_downloaded:
        # Extract
        print("Extracting WordSim-353...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                zip_ref.extractall(dest_dir)
            
            # Find the combined.csv file (might be in a subdirectory)
            if not dest_path.exists():
                # Search for combined.csv in the extracted directory
                for file_path in dest_dir.rglob("combined.csv"):
                    if file_path != dest_path:
                        file_path.rename(dest_path)
                        break
            
            if dest_path.exists():
                return dest_path
        except Exception as e:
            print(f"Failed to extract zip file: {e}")
            last_error = e
    
    # If zip download/extraction failed, try direct CSV download
    print("Trying direct CSV download...")
    for url in csv_urls:
        try:
            print(f"Trying to download WordSim-353 CSV from {url}...")
            download_file(url, dest_path, "WordSim-353 CSV")
            if dest_path.exists():
                return dest_path
        except Exception as e:
            last_error = e
            if dest_path.exists():
                dest_path.unlink()
            continue
    
    # All attempts failed
    raise ConnectionError(
        f"Failed to download WordSim-353 from all attempted sources. "
        f"Last error: {last_error}. "
        f"Please check your internet connection or download manually from:\n"
        f"  - Kaggle: https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd\n"
        f"  - http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip"
    )


def download_rarewords() -> Path:
    """Download Stanford Rare Words dataset."""
    # Rare Words dataset is available as a zip file
    url = "http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip"
    zip_path = CACHE_DIR / "rw.zip"
    dest_path = CACHE_DIR / "rarewords.txt"
    
    if dest_path.exists():
        print(f"Rare Words already exists: {dest_path}")
        return dest_path
    
    # Download zip file
    download_file(url, zip_path, "Stanford Rare Words")
    
    # Extract and find the rw.txt file
    print("Extracting Stanford Rare Words...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        # Extract to a temporary directory
        extract_dir = CACHE_DIR / "rw_extract"
        extract_dir.mkdir(exist_ok=True)
        zip_ref.extractall(extract_dir)
        
        # Find rw.txt (usually in rw/rw.txt)
        rw_txt = None
        for file_path in extract_dir.rglob("rw.txt"):
            rw_txt = file_path
            break
        
        if rw_txt and rw_txt.exists():
            # Copy to destination
            shutil.copy2(rw_txt, dest_path)
            # Clean up extract directory
            shutil.rmtree(extract_dir, ignore_errors=True)
        else:
            raise FileNotFoundError(
                f"Could not find rw.txt after extraction. "
                f"Files in zip: {file_list}"
            )
    
    # Clean up zip file
    if zip_path.exists():
        zip_path.unlink()
    
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

