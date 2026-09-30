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
-- Values above the maximum representable value are clamped to +32767.
-- Values below the minimum representable value are clamped to -32768.
--
-- Implementation detail: this avoids wraparound caused by direct truncation.
--
-- =============================================================================

entity saturate is
    port (
        data_in  : in  accumulator_t;
        data_out : out data_t
    );
end entity saturate;

architecture rtl of saturate is

    -- Maximum signed DATA_WIDTH value
    constant MAX_VALUE : accumulator_t := to_signed((2 ** (DATA_WIDTH - 1)) - 1,ACCUMULATOR_WIDTH);
    -- Minimum signed DATA_WIDTH value
    constant MIN_VALUE : accumulator_t := to_signed(-(2 ** (DATA_WIDTH - 1)),ACCUMULATOR_WIDTH);

begin

    -- There is no problem in using `all` for the sensitivity list because the
    -- parameters other than `data_in` are all constant
    process(all)
    begin

        if data_in > MAX_VALUE then
            data_out <= to_signed((2 ** (DATA_WIDTH - 1)) - 1,DATA_WIDTH);
        elsif data_in < MIN_VALUE then
            data_out <= to_signed(-(2 ** (DATA_WIDTH - 1)),DATA_WIDTH);
        else
            data_out <= resize(data_in,DATA_WIDTH);
        end if;

    end process;

end architecture rtl;