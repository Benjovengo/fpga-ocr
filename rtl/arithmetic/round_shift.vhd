library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

-- =============================================================================
-- Fixed-Point Rescaling
-- =============================================================================
--
-- The quantized weights use 14 fractional bits.
-- After multiplication, the result contains additional fractional bits.
--
-- This block restores the expected scaling by shifting the accumulated value
-- right by FRACTIONAL_BITS.
--
-- This first implementation performs truncation.
-- Proper round-to-nearest can be added later.
--        ----------------
--
-- =============================================================================

entity round_shift is
    port (
        accumulator_in : in  accumulator_t;
        shifted_out     : out accumulator_t
    );
end entity round_shift;

architecture rtl of round_shift is
begin

    -- Arithmetic shift preserves the sign of signed values
    shifted_out <= shift_right(accumulator_in,FRACTIONAL_BITS);

end architecture rtl;