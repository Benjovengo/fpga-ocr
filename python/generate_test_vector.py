#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Project Paths
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_FILE = DATA_DIR / "train.csv"

INPUT_IMAGE_FILE = DATA_DIR / "input_image.mem"
EXPECTED_CLASS_FILE = DATA_DIR / "expected_class.txt"


# =============================================================================
# Configuration
# =============================================================================

# Select which image from train.csv will be simulated.
IMAGE_INDEX = 0


# =============================================================================
# Generate Test Vector
# =============================================================================

def main():

    print(f"Dataset        : {TRAIN_FILE}")
    print(f"Image index    : {IMAGE_INDEX}")

    # Load the MNIST CSV dataset.
    dataset = pd.read_csv(TRAIN_FILE)

    # Select one image.
    sample = dataset.iloc[IMAGE_INDEX]

    # The label column contains the expected MNIST digit.
    expected_class = int(sample["label"])

    # Remove the label and obtain the 784 image pixels.
    pixels = sample.drop("label").to_numpy()

    if len(pixels) != 784:
        raise ValueError(
            f"Expected 784 pixels, found {len(pixels)}"
        )

    # Use exactly the same binarization as training.py:
    #
    #     pixel <= 127 -> 0
    #     pixel >  127 -> 1
    binary_pixels = (
        pixels > 127
    ).astype(np.uint8)

    # Write one binary pixel per line.
    with open(INPUT_IMAGE_FILE, "w") as file:

        for pixel in binary_pixels:

            file.write(f"{pixel}\n")

    # Write the expected classification.
    with open(EXPECTED_CLASS_FILE, "w") as file:

        file.write(f"{expected_class}\n")

    print(f"Expected class : {expected_class}")
    print(f"Input file     : {INPUT_IMAGE_FILE}")
    print(f"Expected file  : {EXPECTED_CLASS_FILE}")
    print(f"Pixels written : {len(binary_pixels)}")


if __name__ == "__main__":
    main()