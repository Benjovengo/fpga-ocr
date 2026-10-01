library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

-- =============================================================================
-- Parallel Neural-Network Weight ROM
-- =============================================================================
--
-- Purpose
-- -------
--
-- Stores one complete flattened neural-network weight matrix and exposes all
-- weights associated with the currently selected input activation in parallel.
--
-- The flattened matrix ordering is:
--
--     address = input_index * OUTPUT_COUNT + output_index
--
-- Example for a layer with 64 outputs:
--
--     input_index = 3
--
--     weights(0)  = ROM(3 * 64 + 0)
--     weights(1)  = ROM(3 * 64 + 1)
--     ...
--     weights(63) = ROM(3 * 64 + 63)
--
-- Quartus Synthesis
-- -----------------
--
-- Quartus initializes the inferred ROM through the ram_init_file attribute.
-- INIT_FILE must point to the Quartus-format MIF for the current layer.
--
-- Questa RTL Simulation
-- ---------------------
--
-- A plain VHDL RTL simulation does not automatically initialize a VHDL array
-- merely because the Quartus ram_init_file attribute exists.
--
-- Therefore, a simulation-only process reads the same MIF from SIM_INIT_FILE.
--
-- No nn_weights_pkg is used.
-- No std.textio package is used.
-- No altera_mf library or altsyncram instance is used.
-- Only IEEE packages and project work packages are imported.
--
-- Expected MIF format:
--
--     WIDTH=16;
--     DEPTH=...;
--     ADDRESS_RADIX=UNS;
--     DATA_RADIX=HEX;
--     CONTENT BEGIN
--     0 : 0BAD;
--     1 : EF8B;
--     ...
--     END;
--
-- =============================================================================

entity weight_rom is
    generic (
        -- Number of input activations in this layer.
        INPUT_COUNT : positive;
        -- Number of output neurons in this layer.
        OUTPUT_COUNT : positive;
        -- MIF path used by Quartus synthesis.
        INIT_FILE : string;
        -- Path to the same MIF used by Questa RTL simulation.
        SIM_INIT_FILE : string
    );
    port (
        -- Input row of the flattened weight matrix.
        input_index : in natural range 0 to INPUT_COUNT - 1;
        -- All weights connected to the selected input activation.
        weights : out weight_array_t(0 to OUTPUT_COUNT - 1)
    );
end entity weight_rom;

architecture rtl of weight_rom is

    -- =========================================================================
    -- ROM Geometry
    -- =========================================================================
    constant DEPTH : positive := INPUT_COUNT * OUTPUT_COUNT;

    -- =========================================================================
    -- ROM Type
    -- =========================================================================
    type rom_t is array (0 to DEPTH - 1) of weight_t;

    -- =========================================================================
    -- Hexadecimal Character Conversion
    -- =========================================================================
    --
    -- Converts one hexadecimal character to its integer value.
    --
    -- The function is pure: it depends only on its input argument and performs
    -- no file access or other side effects.
    --
    -- Return values:
    --
    --      0 ... 15    valid hexadecimal digit
    --     -1           not a hexadecimal digit
    --
    function hex_value(value : character) return integer is
    begin
        case value is
            when '0' =>
                return 0;
            when '1' =>
                return 1;
            when '2' =>
                return 2;
            when '3' =>
                return 3;
            when '4' =>
                return 4;
            when '5' =>
                return 5;
            when '6' =>
                return 6;
            when '7' =>
                return 7;
            when '8' =>
                return 8;
            when '9' =>
                return 9;
            when 'A' | 'a' =>
                return 10;
            when 'B' | 'b' =>
                return 11;
            when 'C' | 'c' =>
                return 12;
            when 'D' | 'd' =>
                return 13;
            when 'E' | 'e' =>
                return 14;
            when 'F' | 'f' =>
                return 15;
            when others =>
                return -1;
        end case;
    end function hex_value;

    -- =========================================================================
    -- ROM Storage
    -- =========================================================================
    --
    -- The explicit default removes Quartus warnings about implicit X
    -- initialization.
    --
    -- For synthesis, Quartus applies INIT_FILE through ram_init_file.
    --
    -- For RTL simulation, the simulation-only mif_loader process below
    -- overwrites these default zeros with the contents of SIM_INIT_FILE at
    -- simulation time zero.
    --
    signal rom : rom_t := (others => (others => '0'));

    -- =========================================================================
    -- Quartus ROM Initialization Attribute
    -- =========================================================================
    --
    -- This attribute is interpreted by Quartus during synthesis.
    -- It is not a generic VHDL mechanism for loading a MIF during RTL
    -- simulation; that is why the separate mif_loader process is provided.
    --
    attribute ram_init_file : string;
    attribute ram_init_file of rom : signal is INIT_FILE;

begin
    -- =========================================================================
    -- Questa RTL-Simulation MIF Loader
    -- =========================================================================
    --
    -- This process is excluded from synthesis.
    --
    -- It deliberately does not use:
    --
    --     file_open_status
    --     open_ok
    --     file_open(...)
    --     std.textio
    --
    -- The file is opened directly by its declaration, avoiding the undefined
    -- file_open_status/open_ok identifiers seen in the previous version.
    --
    -- synthesis translate_off
    mif_loader : process
        -- Direct VHDL file of characters
        type character_file_t is file of character;
        
        -- Open the MIF automatically when this process is elaborated
        -- No file_open_status or open_ok object is required
        file mif_file : character_file_t open read_mode is SIM_INIT_FILE;

        -- Character currently read from the MIF
        variable current_character : character;
        -- Decimal address parsed before ':'
        variable address_value : natural := 0;
        -- Raw hexadecimal data parsed after ':' and before ';'
        variable data_value : natural := 0;
        -- Numeric value corresponding to one hexadecimal character
        variable hexadecimal_digit : integer := -1;
        -- Indicates that at least one decimal address digit has been parsed
        variable address_valid : boolean := false;
        -- Indicates that at least one hexadecimal data digit has been parsed
        variable data_valid : boolean := false;
        -- False while parsing the address side of an entry
        -- True while parsing the hexadecimal data side
        variable reading_data : boolean := false;
        -- Number of MIF words successfully copied into the ROM.
        variable loaded_words : natural := 0;

    begin
        -- ---------------------------------------------------------------------
        -- Parse Entire MIF
        -- ---------------------------------------------------------------------
        --
        -- Header lines are ignored.
        -- A valid data entry begins when decimal characters are followed by
        -- ':' and ends when hexadecimal data is followed by ';'.
        --
        while not endfile(mif_file) loop
            read(mif_file,current_character);

            if not reading_data then
                -- =============================================================
                -- Address Side
                -- =============================================================
                if current_character >= '0' and current_character <= '9' then

                    -- Build a decimal address one character at a time.
                    address_value := address_value * 10 + character'pos(current_character) - character'pos('0');
                    address_valid := true;

                elsif current_character = ':' and address_valid then

                    -- A valid decimal address followed by ':' marks the start
                    -- of hexadecimal weight data.
                    reading_data := true;
                    data_value := 0;
                    data_valid := false;

                elsif current_character = character'val(10) or current_character = character'val(13) then
                    -- Reset digits that may have appeared in header fields
                    -- such as WIDTH=16 or DEPTH=50176.
                    address_value := 0;
                    address_valid := false;
                end if;
            else
                -- =============================================================
                -- Data Side
                -- =============================================================
                hexadecimal_digit := hex_value(current_character);

                if hexadecimal_digit >= 0 then
                    -- Build the hexadecimal value one nibble at a time.
                    data_value := data_value * 16 + natural(hexadecimal_digit);
                    data_valid := true;
                elsif current_character = ';' then
                    -- =========================================================
                    -- Complete MIF Entry
                    -- =========================================================
                    if address_valid and data_valid then
                        assert address_value < DEPTH
                            report
                                "weight_rom: MIF address exceeds ROM depth in " & SIM_INIT_FILE
                            severity failure;

                        if address_value < DEPTH then
                            -- Copy the raw WEIGHT_WIDTH-bit hexadecimal pattern
                            -- into weight_t while preserving two's-complement
                            -- representation.
                            rom(address_value) <= signed(to_unsigned(data_value,WEIGHT_WIDTH));
                            loaded_words := loaded_words + 1;
                        end if;
                    end if;

                    -- Reset the parser for the next MIF entry.
                    address_value := 0;
                    data_value := 0;
                    address_valid := false;
                    data_valid := false;
                    reading_data := false;
                end if;
            end if;
        end loop;


        -- ---------------------------------------------------------------------
        -- Validate Number of Loaded Weights
        -- ---------------------------------------------------------------------

        assert loaded_words = DEPTH
            report
                "weight_rom: loaded "
                & integer'image(loaded_words)
                & " MIF entries from "
                & SIM_INIT_FILE
                & ", expected "
                & integer'image(DEPTH)
            severity failure;

        -- ---------------------------------------------------------------------
        -- Simulation Confirmation
        -- ---------------------------------------------------------------------

        report
            "weight_rom: loaded " & integer'image(loaded_words) & " weights from " & SIM_INIT_FILE severity note;

        -- ---------------------------------------------------------------------
        -- Execute Once
        -- ---------------------------------------------------------------------
        wait;
    end process mif_loader;

    -- synthesis translate_on

    -- =========================================================================
    -- Parallel Weight Read
    -- =========================================================================
    --
    -- For the selected input_index, expose every weight feeding the layer's
    -- output neurons in parallel.
    --
    -- Address mapping:
    --
    --     address = input_index * OUTPUT_COUNT + output_index
    --
    weight_read_gen : for output_index in 0 to OUTPUT_COUNT - 1 generate
        weights(output_index) <= rom(input_index * OUTPUT_COUNT + output_index);
    end generate weight_read_gen;

end architecture rtl;
