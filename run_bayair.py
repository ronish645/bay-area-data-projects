"""Convenient source-checkout entry point for VS Code; install dependencies first."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Bay_Area_Health_and_Pollution_Analysis" / "src"))

from bayair.cli import main

if __name__ == "__main__":
    main()
