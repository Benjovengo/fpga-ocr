library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

-- =============================================================================
-- Signed Saturation
-- =============================================================================
--
-- Converts the wide accumulator value to DATA_WIDTH bits.
--
-- The accumulator is wider than the activation data so that multiple
-- multiplication results can be accumulated without immediately overflowing.
--
-- For DATA_WIDTH = 16:
--
--     maximum =  32767
--     minimum = -32768
--
-- Values above or below this range are saturated instead of being truncated.
-- This prevents two's-complement wraparound.
--
-- `enable` prevents numeric comparisons while the dense-layer accumulator is
-- not yet valid for an output neuron. This keeps RTL simulation free of the
-- repeated NUMERIC_STD metavalue warnings that otherwise occur while ROM data
-- or intermediate signals are still settling.
--
-- =============================================================================

entity saturate is
    port (
        enable   : in  std_logic;
        data_in  : in  accumulator_t;
        data_out : out data_t
    );
end entity saturate;

architecture rtl of saturate is

    -- Wide limits are used for comparisons against `accumulator_t`
    constant MAX_ACC_VALUE : accumulator_t := to_signed((2 ** (DATA_WIDTH - 1)) - 1,ACCUMULATOR_WIDTH);
    constant MIN_ACC_VALUE : accumulator_t := to_signed(-(2 ** (DATA_WIDTH - 1)),ACCUMULATOR_WIDTH);
    -- `DATA_WIDTH` limits are used for the actual saturated output values
    constant MAX_DATA_VALUE : data_t := to_signed((2 ** (DATA_WIDTH - 1)) - 1,DATA_WIDTH);
    constant MIN_DATA_VALUE : data_t := to_signed(-(2 ** (DATA_WIDTH - 1)),DATA_WIDTH);

begin

    process(all)
    begin
        -- Do not evaluate signed comparisons until dense_layer indicates that
        -- the completed accumulator value is ready for post-processing.
        if enable = '0' then
            data_out <= (others => '0');
        elsif is_x(std_logic_vector(data_in)) then
            data_out <= (others => '0');
        elsif data_in > MAX_ACC_VALUE then
            data_out <= MAX_DATA_VALUE;
        elsif data_in < MIN_ACC_VALUE then
            data_out <= MIN_DATA_VALUE;
        else
            data_out <= resize(data_in,DATA_WIDTH);
        end if;
    end process;

end architecture rtl;
