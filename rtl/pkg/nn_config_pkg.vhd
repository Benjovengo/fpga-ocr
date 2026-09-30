library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- =============================================================================
-- Neural-Network Configuration Package
-- =============================================================================
--
-- This package contains the topology and fixed-point parameters shared by the
-- complete neural-network RTL implementation.
--
-- Keeping these constants in one package avoids duplicating values such as
-- 784, 64, 32, and 10 in several RTL files.
--
-- Network topology:
--
--     784 inputs
--         |
--         v
--     Layer 1: 784 -> 64
--         |
--        ReLU
--         |
--         v
--     Layer 2: 64 -> 64
--         |
--        ReLU
--         |
--         v
--     Layer 3: 64 -> 32
--         |
--        ReLU
--         |
--         v
--     Layer 4: 32 -> 10
--         |
--         v
--       ArgMax
--
-- Weight representation:
--
--     signed 16-bit
--     14 fractional bits
--     scale = 2^14 = 16384
--
-- =============================================================================

package nn_config_pkg is

    -- Number of pixels in one flattened 28 x 28 MNIST image.
    constant INPUT_SIZE : positive := 784;

    -- -------------------------------------------------------------------------
    -- Layer 1
    -- -------------------------------------------------------------------------
    constant LAYER1_INPUTS  : positive := 784;
    constant LAYER1_OUTPUTS : positive := 64;

    -- -------------------------------------------------------------------------
    -- Layer 2
    -- -------------------------------------------------------------------------
    constant LAYER2_INPUTS  : positive := 64;
    constant LAYER2_OUTPUTS : positive := 64;

    -- -------------------------------------------------------------------------
    -- Layer 3
    -- -------------------------------------------------------------------------
    constant LAYER3_INPUTS  : positive := 64;
    constant LAYER3_OUTPUTS : positive := 32;

    -- -------------------------------------------------------------------------
    -- Layer 4 / Output Layer
    -- -------------------------------------------------------------------------
    constant LAYER4_INPUTS  : positive := 32;
    constant LAYER4_OUTPUTS : positive := 10;

    -- MNIST contains ten classes: digits 0 through 9.
    constant NUM_CLASSES : positive := 10;

    -- -------------------------------------------------------------------------
    -- Fixed-Point Configuration
    -- -------------------------------------------------------------------------

    -- Width of each stored activation.
    constant DATA_WIDTH : positive := 16;
    -- Width of each quantized weight.
    constant WEIGHT_WIDTH : positive := 16;
    -- Number of bits after the binary point.
    constant FRACTIONAL_BITS : natural := 14;
    -- Fixed-point scaling factor.
    --
    --     2^14 = 16384
    constant QUANTIZATION_SCALE : positive := 16384;

    -- -------------------------------------------------------------------------
    -- Accumulator
    -- -------------------------------------------------------------------------

    -- The accumulator must be wider than the input and weight values.
    --
    -- A 16-bit input multiplied by a 16-bit weight produces up to 32 bits.
    --
    -- Extra bits are required because hundreds of multiplication results may
    -- be accumulated for a single neuron.
    constant ACCUMULATOR_WIDTH : positive := 48;

end package nn_config_pkg;