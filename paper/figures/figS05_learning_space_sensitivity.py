"""Old-reference sensitivity of the controlled learning-space comparison."""
import argparse
from pathlib import Path

from paper.figures.fig02_learning_space import make


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir, reference="old675LR56")
