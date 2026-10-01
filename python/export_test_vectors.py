#!/usr/bin/env python3

# =============================================================================
# FPGA OCR Test-Vector Exporter
# =============================================================================
#
# This script creates the files consumed by the Verilog RTL testbench:
#
#     data/test_images.mem
#     data/test_labels.mem
#
# Every MNIST image contains:
#
#     28 x 28 = 784 pixels
#
# training.py binarizes the MNIST pixels using:
#
#     pixel <= 127 -> 0
#     pixel >  127 -> 1
#
# The FPGA input_buffer uses exactly the same representation.
#
# test_images.mem therefore contains one binary pixel per line.
#
# Images are concatenated sequentially:
#
#     image 0 -> lines    0 ... 783
#     image 1 -> lines  784 ... 1567
#     image 2 -> lines 1568 ... 2351
#     ...
#
# The Verilog testbench recovers a pixel using:
#
#     memory_index =
#         image_index * INPUT_PIXELS
#         + pixel_index;
#
# test_labels.mem contains one expected decimal digit per line.
#
# =============================================================================

from pathlib import Path

import torch
from torchvision import datasets
from torchvision import transforms


# =============================================================================
# Project Paths
# =============================================================================

# Resolve the project root from:
#
#     project/python/export_test_vectors.py
#
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Existing MNIST directory
DATA_DIR = PROJECT_DIR / "data"
# Directory containing generated RTL/testbench configuration files
CONFIG_DIR = PROJECT_DIR / "config"

# Output files consumed by nn_core_tb.v
TEST_IMAGES_FILE = DATA_DIR / "test_images.mem"
TEST_LABELS_FILE = DATA_DIR / "test_labels.mem"
# Verilog configuration generated together with the test vectors
TEST_CONFIG_FILE = CONFIG_DIR / "test_vectors.vh"


# =============================================================================
# Test Configuration
# =============================================================================

# MNIST image dimensions
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

# Number of pixels presented to the FPGA for each image
INPUT_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

# Pixel threshold must match training.py.
BINARIZATION_THRESHOLD = 127

# Debug: start with a small number while debugging RTL.
#
# Increase later to:
#
#     100
#     1000
#     10000
#
NUM_TEST_IMAGES = 100
# Fraction of the available MNIST test dataset exported for RTL verification.
TEST_FRACTION = 0.20


# =============================================================================
# Load MNIST Test Dataset
# =============================================================================

def load_test_dataset():
    """
    Load the official MNIST test dataset.

    The dataset is expected to already exist below:

        data/MNIST/

    download=False prevents this RTL-vector generation step from unexpectedly
    downloading or modifying the dataset.
    """

    dataset = datasets.MNIST(root=DATA_DIR,train=False,download=False,transform=transforms.ToTensor(),)
    print(f"MNIST test samples : {len(dataset)}")

    return dataset


# =============================================================================
# Convert Image to FPGA Binary Representation
# =============================================================================

def binarize_image(image):
    """
    Convert one MNIST image to the exact one-bit representation expected by
    input_buffer.vhd.

    torchvision ToTensor() converts pixels from:

        0 ... 255

    to:

        0.0 ... 1.0

    Therefore the threshold 127 corresponds to:

        127 / 255

    in the tensor representation.

    Returns a flattened tensor containing exactly 784 integer values:

        0 or 1
    """

    normalized_threshold = (BINARIZATION_THRESHOLD / 255.0)

    binary_image = (image > normalized_threshold).to(torch.uint8)

    # Convert:
    #
    #     [1, 28, 28]
    #
    # to:
    #
    #     [784]
    #
    flattened_image = binary_image.flatten()

    if flattened_image.numel() != INPUT_PIXELS:
        raise ValueError(f"Expected {INPUT_PIXELS} pixels, got {flattened_image.numel()}")

    return flattened_image


# =============================================================================
# Export Images
# =============================================================================

def export_images(dataset, number_of_images):
    """
    Write all FPGA input pixels to one test_images.mem file.

    One pixel is written per line.

    Example for two images:

        image 0 pixel 0
        image 0 pixel 1
        ...
        image 0 pixel 783
        image 1 pixel 0
        image 1 pixel 1
        ...
        image 1 pixel 783

    No image separators are needed because every image always contains exactly
    INPUT_PIXELS values.
    """

    print(f"Creating image vectors: {TEST_IMAGES_FILE}")

    with TEST_IMAGES_FILE.open("w",encoding="utf-8",) as output_file:
        for image_index in range(number_of_images):
            image, _ = dataset[image_index]
            binary_image = binarize_image(image)
            for pixel in binary_image:
                output_file.write(f"{int(pixel.item())}\n")


# =============================================================================
# Export Labels
# =============================================================================

def export_labels(dataset, number_of_images):
    """
    Write one expected MNIST class per line.

    Example:

        7
        2
        1
        0
        ...

    The Verilog testbench indexes this file using the same image_index used for
    test_images.mem.
    """

    print(f"Creating label vectors: {TEST_LABELS_FILE}")

    with TEST_LABELS_FILE.open("w",encoding="utf-8",) as output_file:
        for image_index in range(number_of_images):
            _, label = dataset[image_index]
            output_file.write(f"{int(label)}\n")

# =============================================================================
# Export Verilog Testbench Configuration
# =============================================================================

def export_testbench_config(number_of_images):
    """
    Generate the Verilog configuration used by nn_core_tb.v.

    The number of images is derived from the actual dataset size and
    TEST_FRACTION, so the Verilog testbench does not need to duplicate this
    information.

    Generated file:

        config/test_vectors.vh

    Example:

        `define NUM_TEST_IMAGES 2000
    """

    # Make sure the configuration directory exists
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)

    print(f"Creating test configuration: {TEST_CONFIG_FILE}")

    with TEST_CONFIG_FILE.open("w",encoding="utf-8",) as output_file:
        output_file.write("// =============================================================\n")
        output_file.write("// FPGA OCR Test-Vector Configuration\n")
        output_file.write("// =============================================================\n")
        output_file.write("//\n")

        output_file.write("// Automatically generated by python/export_test_vectors.py.\n")
        output_file.write("// Do not edit manually.\n")
        output_file.write("//\n\n")

        output_file.write("`ifndef TEST_VECTORS_VH\n")
        output_file.write("`define TEST_VECTORS_VH\n\n")
        output_file.write(f"`define NUM_TEST_IMAGES {number_of_images}\n")
        output_file.write(f"`define TEST_INPUT_SIZE {INPUT_PIXELS}\n")
        output_file.write(f"`define TOTAL_TEST_PIXELS "f"{number_of_images * INPUT_PIXELS}\n\n")

        output_file.write("`endif\n")


# =============================================================================
# Verify Generated Files
# =============================================================================

def verify_files(number_of_images):
    """
    Verify that the generated files contain the expected number of entries.

    test_images.mem must contain:

        number_of_images * 784

    lines.

    test_labels.mem must contain:

        number_of_images

    lines.
    """

    with TEST_IMAGES_FILE.open("r",encoding="utf-8",) as input_file:
        image_lines = sum(1 for _ in input_file)

    with TEST_LABELS_FILE.open("r",encoding="utf-8",) as input_file:
        label_lines = sum(1 for _ in input_file)

    expected_image_lines = (number_of_images * INPUT_PIXELS)

    if image_lines != expected_image_lines:
        raise ValueError(
            f"test_images.mem contains "
            f"{image_lines} values; "
            f"expected {expected_image_lines}"
        )

    if label_lines != number_of_images:
        raise ValueError(f"test_labels.mem contains {label_lines} values; expected {number_of_images}")

    print(f"Image values written : {image_lines}")
    print(f"Labels written       : {label_lines}")


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Generate the complete set of RTL input and expected-output vectors.
    """

    print(f"Project directory : {PROJECT_DIR}")
    print(f"Images file       : {TEST_IMAGES_FILE}")
    print(f"Labels file       : {TEST_LABELS_FILE}")

    dataset = load_test_dataset()
    # Use 20% of the images as test
    number_of_images = int(len(dataset) * TEST_FRACTION)
    # number_of_images = NUM_TEST_IMAGES
    print(f"Images to export  : {number_of_images}")

    if NUM_TEST_IMAGES > len(dataset):
        raise ValueError(f"Requested {number_of_images} images, but the dataset contains only {len(dataset)} samples")

    export_images(dataset,number_of_images)

    export_labels(dataset,number_of_images)

    export_testbench_config(number_of_images)

    verify_files(number_of_images)

    print("")
    print("Test-vector generation completed.")


if __name__ == "__main__":
    main()