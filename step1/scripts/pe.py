"""Run from any working directory: python <project>/scripts/pe.py --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pinkelephant.cli import main

if __name__ == "__main__":
    main()

