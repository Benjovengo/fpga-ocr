#!/usr/bin/env python3

# =============================================================================
# FPGA OCR Fixed-Point RTL Reference Model
# =============================================================================
#
# Purpose
# -------
#
# Reproduce the arithmetic performed by the VHDL neural network using exactly
# the same:
#
#     - input images
#     - expected labels
#     - quantized weights
#     - fixed-point shifts
#     - saturation
#     - ReLU
#     - ArgMax
#
# Files used:
#
#     data/test_images.mem
#     data/test_labels.mem
#
#     model/quartus_mif/matrix1.mif
#     model/quartus_mif/matrix2.mif
#     model/quartus_mif/matrix3.mif
#     model/quartus_mif/matrix4.mif
#
# This script intentionally does NOT load:
#
#     mnist_model.pth
#     mnist_model_quantized.pth
#
# because the purpose is to compare against the actual values consumed by the
# FPGA implementation.
#
# =============================================================================

from pathlib import Path

import numpy as np


# =============================================================================
# Project Paths
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

MIF_DIR = PROJECT_DIR / "model" / "quartus_mif"

TEST_IMAGES_FILE = DATA_DIR / "test_images.mem"
TEST_LABELS_FILE = DATA_DIR / "test_labels.mem"

MATRIX1_FILE = MIF_DIR / "matrix1.mif"
MATRIX2_FILE = MIF_DIR / "matrix2.mif"
MATRIX3_FILE = MIF_DIR / "matrix3.mif"
MATRIX4_FILE = MIF_DIR / "matrix4.mif"


# =============================================================================
# Neural-Network Configuration
# =============================================================================

INPUT_PIXELS = 784

LAYER1_INPUTS = 784
LAYER1_OUTPUTS = 64

LAYER2_INPUTS = 64
LAYER2_OUTPUTS = 64

LAYER3_INPUTS = 64
LAYER3_OUTPUTS = 32

LAYER4_INPUTS = 32
LAYER4_OUTPUTS = 10

NUM_CLASSES = 10


# =============================================================================
# Fixed-Point Configuration
# =============================================================================
#
# The weights and activations use signed 16-bit Q1.14 values.
#
# Layers 2-4 multiply:
#
#     Q1.14 x Q1.14
#
# producing a value with 28 fractional bits.
#
# After accumulation the RTL shifts right by 14 bits.
#
# Layer 1 is special because its inputs are binary 0/1 values. The RTL bypasses
# the multiplier and directly adds the selected Q1.14 weight.
#
# =============================================================================

DATA_WIDTH = 16
WEIGHT_WIDTH = 16
FRACTIONAL_BITS = 14

INT16_MIN = -(1 << (DATA_WIDTH - 1))
INT16_MAX = (1 << (DATA_WIDTH - 1)) - 1


# =============================================================================
# Two's-Complement Conversion
# =============================================================================

def unsigned_to_signed(value, width):
    """
    Interpret an unsigned integer as a width-bit two's-complement value.

    Example for 16 bits:

        0x0001 ->     1
        0x7FFF -> 32767
        0x8000 -> -32768
        0xFFFF ->    -1
    """

    sign_bit = 1 << (width - 1)
    full_scale = 1 << width

    if value & sign_bit:
        return value - full_scale

    return value


# =============================================================================
# Read Quartus MIF
# =============================================================================

def read_mif(file_path, expected_depth):
    """
    Read a Quartus MIF containing entries such as:

        WIDTH=16;
        DEPTH=50176;

        ADDRESS_RADIX=UNS;
        DATA_RADIX=HEX;

        CONTENT BEGIN
        0 : 0001;
        1 : FFFF;
        ...
        END;

    Returns a signed int64 NumPy array.

    int64 is deliberately used for the Python calculation so intermediate
    accumulation does not overflow at int16 boundaries.
    """

    memory = np.zeros(
        expected_depth,
        dtype=np.int64,
    )

    loaded = np.zeros(
        expected_depth,
        dtype=bool,
    )

    inside_content = False

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as mif_file:

        for raw_line in mif_file:

            line = raw_line.strip()

            if not line:
                continue

            upper_line = line.upper()

            if upper_line.startswith("CONTENT BEGIN"):
                inside_content = True
                continue

            if upper_line == "END;":
                break

            if not inside_content:
                continue

            if ":" not in line:
                continue

            address_text, data_text = line.split(
                ":",
                maxsplit=1,
            )

            address_text = address_text.strip()

            data_text = (
                data_text
                .strip()
                .rstrip(";")
                .strip()
            )

            # This reference model expects the format currently generated for
            # the RTL project: explicit decimal addresses and hexadecimal data.
            address = int(
                address_text,
                10,
            )

            raw_value = int(
                data_text,
                16,
            )

            if address < 0 or address >= expected_depth:
                raise ValueError(
                    f"{file_path}: address {address} "
                    f"is outside expected depth "
                    f"{expected_depth}"
                )

            memory[address] = unsigned_to_signed(
                raw_value,
                WEIGHT_WIDTH,
            )

            loaded[address] = True

    loaded_count = int(
        np.count_nonzero(loaded)
    )

    if loaded_count != expected_depth:
        raise ValueError(
            f"{file_path}: loaded {loaded_count} weights, "
            f"expected {expected_depth}"
        )

    print(
        f"{file_path.name:12s}: "
        f"{loaded_count:6d} weights"
    )

    return memory


# =============================================================================
# Load Weight Matrices
# =============================================================================

def load_weights():
    """
    Load the same four MIF matrices used by the FPGA.

    IMPORTANT

    The current VHDL weight-ROM addressing is:

        address =
            input_index * OUTPUT_COUNT
            + output_index

    Therefore each flattened MIF is reshaped as:

        [input][output]

    and NOT:

        [output][input]
    """

    matrix1_flat = read_mif(
        MATRIX1_FILE,
        LAYER1_INPUTS * LAYER1_OUTPUTS,
    )

    matrix2_flat = read_mif(
        MATRIX2_FILE,
        LAYER2_INPUTS * LAYER2_OUTPUTS,
    )

    matrix3_flat = read_mif(
        MATRIX3_FILE,
        LAYER3_INPUTS * LAYER3_OUTPUTS,
    )

    matrix4_flat = read_mif(
        MATRIX4_FILE,
        LAYER4_INPUTS * LAYER4_OUTPUTS,
    )

    matrix1 = matrix1_flat.reshape(
        LAYER1_INPUTS,
        LAYER1_OUTPUTS,
    )

    matrix2 = matrix2_flat.reshape(
        LAYER2_INPUTS,
        LAYER2_OUTPUTS,
    )

    matrix3 = matrix3_flat.reshape(
        LAYER3_INPUTS,
        LAYER3_OUTPUTS,
    )

    matrix4 = matrix4_flat.reshape(
        LAYER4_INPUTS,
        LAYER4_OUTPUTS,
    )

    return (
        matrix1,
        matrix2,
        matrix3,
        matrix4,
    )


# =============================================================================
# Load FPGA Test Images
# =============================================================================

def load_test_images():
    """
    Load exactly the same already-binarized pixels consumed by nn_core_tb.v.

    test_images.mem contains one 0/1 pixel per line.

    Every 784 values form one MNIST image.
    """

    pixels = []

    with TEST_IMAGES_FILE.open(
        "r",
        encoding="utf-8",
    ) as input_file:

        for line_number, raw_line in enumerate(
            input_file,
            start=1,
        ):

            line = raw_line.strip()

            if not line:
                continue

            if line not in ("0", "1"):
                raise ValueError(
                    f"{TEST_IMAGES_FILE}: "
                    f"invalid binary pixel '{line}' "
                    f"at line {line_number}"
                )

            pixels.append(
                int(line)
            )

    pixels = np.asarray(
        pixels,
        dtype=np.int64,
    )

    if len(pixels) % INPUT_PIXELS != 0:
        raise ValueError(
            f"{TEST_IMAGES_FILE} contains "
            f"{len(pixels)} pixels, which is not "
            f"divisible by {INPUT_PIXELS}"
        )

    images = pixels.reshape(
        -1,
        INPUT_PIXELS,
    )

    return images


# =============================================================================
# Load Expected Labels
# =============================================================================

def load_test_labels():
    """
    Load exactly the same labels consumed by the Verilog testbench.

    The file was written for $readmemh, but decimal digits 0-9 have identical
    numeric values when interpreted as one hexadecimal digit.
    """

    labels = []

    with TEST_LABELS_FILE.open(
        "r",
        encoding="utf-8",
    ) as input_file:

        for line_number, raw_line in enumerate(
            input_file,
            start=1,
        ):

            line = raw_line.strip()

            if not line:
                continue

            value = int(
                line,
                16,
            )

            if value < 0 or value >= NUM_CLASSES:
                raise ValueError(
                    f"{TEST_LABELS_FILE}: "
                    f"invalid label {value} "
                    f"at line {line_number}"
                )

            labels.append(
                value
            )

    return np.asarray(
        labels,
        dtype=np.int64,
    )


# =============================================================================
# RTL Saturation
# =============================================================================

def saturate_int16(values):
    """
    Reproduce saturate.vhd.

    Values greater than +32767 become +32767.

    Values less than -32768 become -32768.
    """

    return np.clip(
        values,
        INT16_MIN,
        INT16_MAX,
    ).astype(np.int64)


# =============================================================================
# RTL ReLU
# =============================================================================

def relu(values):
    """
    Reproduce relu.vhd.

        x < 0 -> 0
        x >= 0 -> x
    """

    return np.maximum(
        values,
        0,
    ).astype(np.int64)


# =============================================================================
# Layer 1
# =============================================================================

def layer1_forward(
    image,
    weights,
):
    """
    Reproduce the Layer-1 binary optimization.

    RTL behavior for each input pixel:

        pixel = 0:
            contribution = 0

        pixel = 1:
            contribution = weight

    Because the input itself is binary rather than Q1.14, Layer 1 does NOT
    perform the 14-bit post-accumulation shift used by Layers 2-4.
    """

    accumulators = np.zeros(
        LAYER1_OUTPUTS,
        dtype=np.int64,
    )

    # Deliberately retain the same conceptual ordering as the RTL:
    #
    #     one input per iteration
    #     all output neurons updated in parallel
    #
    for input_index in range(LAYER1_INPUTS):

        if image[input_index] == 1:

            accumulators += weights[
                input_index,
                :
            ]

    saturated = saturate_int16(
        accumulators
    )

    outputs = relu(
        saturated
    )

    return outputs


# =============================================================================
# Generic Fixed-Point Dense + ReLU Layer
# =============================================================================

def dense_relu_forward(
    inputs,
    weights,
):
    """
    Reproduce Layers 2 and 3.

    For every input activation:

        product =
            input_data * weight

        accumulator += product

    After all inputs:

        shifted =
            accumulator >> FRACTIONAL_BITS

        saturated =
            saturate_int16(shifted)

        output =
            ReLU(saturated)

    NumPy/Python right shift on signed int64 values is arithmetic, matching the
    signed shift_right operation used by the VHDL implementation.
    """

    output_count = weights.shape[1]

    accumulators = np.zeros(
        output_count,
        dtype=np.int64,
    )

    for input_index in range(
        len(inputs)
    ):

        products = (
            np.int64(inputs[input_index])
            * weights[input_index, :]
        )

        accumulators += products

    shifted = (
        accumulators
        >> FRACTIONAL_BITS
    )

    saturated = saturate_int16(
        shifted
    )

    outputs = relu(
        saturated
    )

    return outputs


# =============================================================================
# Layer 4
# =============================================================================

def layer4_forward(
    inputs,
    weights,
):
    """
    Reproduce the current Layer-4 RTL.

    Layer 4 performs the fixed-point multiplication and accumulation exactly
    like Layers 2 and 3.

    It then shifts right by FRACTIONAL_BITS.

    IMPORTANT:

    The current VHDL passes dense_output_logit_data to layer4_buffer before
    ArgMax. Therefore ArgMax sees the WIDE shifted logits rather than saturated
    int16 output_data.

    Do NOT apply ReLU here.
    Do NOT saturate these logits before ArgMax.
    """

    accumulators = np.zeros(
        LAYER4_OUTPUTS,
        dtype=np.int64,
    )

    for input_index in range(
        LAYER4_INPUTS
    ):

        products = (
            np.int64(inputs[input_index])
            * weights[input_index, :]
        )

        accumulators += products

    logits = (
        accumulators
        >> FRACTIONAL_BITS
    )

    return logits


# =============================================================================
# Complete RTL-Compatible Inference
# =============================================================================

def predict_image(
    image,
    matrix1,
    matrix2,
    matrix3,
    matrix4,
):
    """
    Execute the complete four-layer fixed-point neural network.
    """

    layer1 = layer1_forward(
        image,
        matrix1,
    )

    layer2 = dense_relu_forward(
        layer1,
        matrix2,
    )

    layer3 = dense_relu_forward(
        layer2,
        matrix3,
    )

    logits = layer4_forward(
        layer3,
        matrix4,
    )

    # np.argmax returns the first maximum.
    #
    # That matches the current ArgMax implementation because the VHDL updates
    # max_index only when:
    #
    #     values(index) > max_value
    #
    # rather than >=.
    predicted_digit = int(
        np.argmax(logits)
    )

    return (
        predicted_digit,
        logits,
    )


# =============================================================================
# Main Verification
# =============================================================================

def main():

    print("")
    print("========================================")
    print(" FPGA OCR PYTHON FIXED-POINT REFERENCE")
    print("========================================")
    print("")

    # -------------------------------------------------------------------------
    # Load exactly the FPGA's weights.
    # -------------------------------------------------------------------------

    print("Loading MIF matrices...")

    (
        matrix1,
        matrix2,
        matrix3,
        matrix4,
    ) = load_weights()

    print("")

    # -------------------------------------------------------------------------
    # Load exactly the FPGA's input vectors.
    # -------------------------------------------------------------------------

    images = load_test_images()
    labels = load_test_labels()

    if len(images) != len(labels):
        raise ValueError(
            f"Image count ({len(images)}) "
            f"does not match label count "
            f"({len(labels)})"
        )

    number_of_images = len(images)

    print(
        f"Images loaded      : "
        f"{number_of_images}"
    )

    print("")

    # -------------------------------------------------------------------------
    # Run Fixed-Point Inference
    # -------------------------------------------------------------------------

    correct = 0
    incorrect = 0

    predictions = []

    for image_index in range(
        number_of_images
    ):

        predicted_digit, logits = predict_image(
            images[image_index],
            matrix1,
            matrix2,
            matrix3,
            matrix4,
        )

        expected_digit = int(
            labels[image_index]
        )

        predictions.append(
            predicted_digit
        )

        if predicted_digit == expected_digit:

            correct += 1

        else:

            incorrect += 1

            # Print a limited amount of diagnostic information for incorrect
            # classifications.
            if incorrect <= 20:

                print(
                    f"Image {image_index}: "
                    f"FAIL "
                    f"expected={expected_digit} "
                    f"predicted={predicted_digit}"
                )

                print(
                    f"  logits = "
                    f"{logits.tolist()}"
                )

    accuracy = (
        100.0
        * correct
        / number_of_images
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("")
    print("========================================")
    print(" PYTHON FIXED-POINT TEST SUMMARY")
    print("========================================")

    print(
        f"Images tested       : "
        f"{number_of_images}"
    )

    print(
        f"Correct             : "
        f"{correct}"
    )

    print(
        f"Incorrect           : "
        f"{incorrect}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.2f} %"
    )

    print("========================================")
    print("")

    # -------------------------------------------------------------------------
    # FPGA Result Used for Comparison
    # -------------------------------------------------------------------------

    fpga_correct = 1261
    fpga_incorrect = 739
    fpga_accuracy = 63.05

    print("========================================")
    print(" FPGA vs PYTHON")
    print("========================================")

    print(
        f"{'Metric':20s} "
        f"{'FPGA':>12s} "
        f"{'Python':>12s}"
    )

    print(
        f"{'Correct':20s} "
        f"{fpga_correct:12d} "
        f"{correct:12d}"
    )

    print(
        f"{'Incorrect':20s} "
        f"{fpga_incorrect:12d} "
        f"{incorrect:12d}"
    )

    print(
        f"{'Accuracy':20s} "
        f"{fpga_accuracy:11.2f}% "
        f"{accuracy:11.2f}%"
    )

    print("========================================")

    # -------------------------------------------------------------------------
    # Exact Aggregate Match
    # -------------------------------------------------------------------------

    if (
        correct == fpga_correct
        and
        incorrect == fpga_incorrect
    ):

        print("")
        print(
            "PASS: Python and FPGA aggregate "
            "classification results match."
        )

    else:

        print("")
        print(
            "FAIL: Python and FPGA aggregate "
            "classification results differ."
        )


if __name__ == "__main__":
    main()