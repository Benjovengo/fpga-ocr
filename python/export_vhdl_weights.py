#!/usr/bin/env python3

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "model"
RTL_PKG_DIR = PROJECT_DIR / "rtl" / "pkg"

OUTPUT_FILE = RTL_PKG_DIR / "nn_weights_pkg.vhd"

MATRIX_FILES = [
    MODEL_DIR / "matrix1_raw_hex.mif",
    MODEL_DIR / "matrix2_raw_hex.mif",
    MODEL_DIR / "matrix3_raw_hex.mif",
    MODEL_DIR / "matrix4_raw_hex.mif",
]


def read_weights(filename):
    with open(filename, "r") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def write_weight_array(file, layer, weights):
    file.write(
        f"    constant LAYER{layer}_WEIGHTS : "
        f"layer{layer}_weights_t := (\n"
    )

    for address, value in enumerate(weights):
        comma = "," if address < len(weights) - 1 else ""

        file.write(
            f'        {address} => signed\'(x"{value}")'
            f"{comma}\n"
        )

    file.write("    );\n\n")


def main():
    matrices = [
        read_weights(filename)
        for filename in MATRIX_FILES
    ]

    expected_depths = [
        784 * 64,
        64 * 64,
        64 * 32,
        32 * 10,
    ]

    for index, (weights, expected) in enumerate(
        zip(matrices, expected_depths),
        start=1,
    ):
        if len(weights) != expected:
            raise ValueError(
                f"Layer {index}: expected {expected} weights, "
                f"found {len(weights)}"
            )

    with open(OUTPUT_FILE, "w") as file:
        file.write(
            """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

package nn_weights_pkg is

    type layer1_weights_t is array (
        0 to LAYER1_INPUTS * LAYER1_OUTPUTS - 1
    ) of weight_t;

    type layer2_weights_t is array (
        0 to LAYER2_INPUTS * LAYER2_OUTPUTS - 1
    ) of weight_t;

    type layer3_weights_t is array (
        0 to LAYER3_INPUTS * LAYER3_OUTPUTS - 1
    ) of weight_t;

    type layer4_weights_t is array (
        0 to LAYER4_INPUTS * LAYER4_OUTPUTS - 1
    ) of weight_t;

"""
        )

        for layer, weights in enumerate(matrices, start=1):
            write_weight_array(
                file,
                layer,
                weights,
            )

        file.write(
            "end package nn_weights_pkg;\n"
        )

    print(f"Created: {OUTPUT_FILE}")

    for layer, weights in enumerate(matrices, start=1):
        print(
            f"Layer {layer} weights: {len(weights)}"
        )


if __name__ == "__main__":
    main()