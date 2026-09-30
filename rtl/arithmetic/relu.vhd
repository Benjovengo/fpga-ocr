library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_types_pkg.all;

-- =============================================================================
-- ReLU Activation
-- =============================================================================
--
-- Implements:
--
--     ReLU(x) = 0, when x < 0
--               x, otherwise
--
-- The sign bit is checked directly because data_t is a signed value.
--
-- =============================================================================

entity relu is
    port (
        data_in  : in  data_t;
        data_out : out data_t
    );
end entity relu;

architecture rtl of relu is
begin

    process(data_in)
    begin

        -- A value with the MSB set is negative in two's complement
        if data_in(data_in'high) = '1' then
            data_out <= (others => '0');
        else
            data_out <= data_in;
        end if;

    end process;

end architecture rtl;