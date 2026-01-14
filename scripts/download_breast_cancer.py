#!/usr/bin/env python3
"""
Download Breast Cancer dataset from Kaggle to data/breast_cancer/
Kaggle: https://www.kaggle.com/datasets/reihanenamdari/breast-cancer [web:13]
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    data_dir = Path("data/breast_cancer")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = data_dir / "Breast_Cancer.csv"
    
    if csv_path.exists():
        print(f"✓ Data already exists: {csv_path}")
        return
    
    print("Downloading Breast Cancer dataset from Kaggle...")
    
    try:
        # Kaggle CLI (pip install kaggle)
        cmd = [
            "kaggle", "datasets", "download", "-d", "reihanenamdari/breast-cancer",
            "-p", str(data_dir), "--unzip", "--force"
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        print(f"✓ Downloaded to: {csv_path}")
    except FileNotFoundError:
        print("Kaggle CLI not found. Install: pip install kaggle")
        print("Setup API token: https://www.kaggle.com/settings/account -> Create New Token")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Download failed. Check internet/Kaggle token.")
        sys.exit(1)

if __name__ == "__main__":
    main()
