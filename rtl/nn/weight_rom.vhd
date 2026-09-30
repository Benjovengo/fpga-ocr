library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;
use work.nn_weights_pkg.all;

-- =============================================================================
-- Weight ROM
-- =============================================================================
--
-- Implements a synchronous neural-network weight ROM using standard VHDL.
--
-- No Intel-specific megafunction is instantiated.
--
-- The ROM contents are provided by nn_weights_pkg.vhd so that the same weights
-- are visible during Questa RTL simulation.
--
-- Quartus can infer this structure as ROM.
--
-- =============================================================================

entity weight_rom is
    generic (
        LAYER_ID : positive;
        DEPTH    : positive
    );
    port (
        clk : in std_logic;
        address : in natural range 0 to DEPTH - 1;
        data_out : out weight_t
    );
end entity weight_rom;

architecture rtl of weight_rom is

    -- Registered ROM output.
    signal data_out_i : weight_t := (others => '0');

    -- Combinational weight selected from the constant array.
    --
    -- This signal is useful for observing the selected package weight in
    -- Questa before the synchronous output register.
    signal selected_weight : weight_t := (others => '0');

begin

    -- Drive the entity output from the registered ROM value.
    data_out <= data_out_i;

    -- =========================================================================
    -- Weight Selection
    -- =========================================================================
    --
    -- LAYER_ID is a generic constant, not a signal.
    --
    -- Therefore it must not be included explicitly in a traditional process
    -- sensitivity list.
    --
    -- VHDL-2008 process(all) automatically includes every signal read by this
    -- combinational process. LAYER_ID remains a constant evaluated during
    -- elaboration.
    process(all)
    begin
        case LAYER_ID is

            when 1 =>
                selected_weight <= LAYER1_WEIGHTS(address);

            when 2 =>
                selected_weight <= LAYER2_WEIGHTS(address);

            when 3 =>
                selected_weight <= LAYER3_WEIGHTS(address);

            when 4 =>
                selected_weight <= LAYER4_WEIGHTS(address);

            when others =>
                selected_weight <= (others => '0');

        end case;
    end process;

    -- =========================================================================
    -- Synchronous ROM Output
    -- =========================================================================
    --
    -- Register the selected weight so that the ROM presents a one-clock
    -- address-to-data latency.
    process(clk)
    begin
        if rising_edge(clk) then
            data_out_i <= selected_weight;
        end if;
    end process;

end architecture rtl;