library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

-- =============================================================================
-- Multiply-Accumulate Unit
-- =============================================================================
--
-- Computes:
--
--     accumulator_out = accumulator_in + data_in * weight_in
--
-- This block is reused sequentially by the dense-layer controller.
--
-- =============================================================================

entity mac is
    port (
        data_in        : in  data_t;
        weight_in      : in  weight_t;
        accumulator_in : in  accumulator_t;
        accumulator_out : out accumulator_t
    );
end entity mac;

architecture rtl of mac is

    -- A multiplication of two signed 16-bit values produces a 32-bit result.
    signal product : signed( DATA_WIDTH + WEIGHT_WIDTH - 1 downto 0);

begin

    -- Perform the signed multiplication.
    product <= data_in * weight_in;
    -- Extend the multiplication result to the accumulator width before adding.
    accumulator_out <= accumulator_in + resize(product, ACCUMULATOR_WIDTH);

end architecture rtl;