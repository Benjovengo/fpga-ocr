library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

-- =============================================================================
-- Weight ROM
-- =============================================================================
--
-- Stores the quantized neural-network weights.
--
-- Quartus can initialize the inferred ROM from the MIF file specified by
-- `INIT_FILE` variable (string).
--
-- =============================================================================

entity weight_rom is
    generic (
        DEPTH     : positive;
        INIT_FILE : string
    );

    port (
        clk : in std_logic;
        address : in natural range 0 to DEPTH - 1;
        data_out : out weight_t
    );
end entity weight_rom;

architecture rtl of weight_rom is

    -- Memory type containing all weights
    type rom_t is array (0 to DEPTH - 1) of weight_t;
    -- The synthesis tool may infer embedded memory from this array
    signal rom : rom_t := (others => (others => '0'));

    -- Quartus-specific initialization
    attribute ram_init_file : string;
    attribute ram_init_file of rom : signal is INIT_FILE;

begin

    -- Synchronous ROM output
    process(clk)
    begin

        if rising_edge(clk) then
            data_out <= rom(address);
        end if;

    end process;

end architecture rtl;