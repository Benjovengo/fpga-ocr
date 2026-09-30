library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

-- =============================================================================
-- Input Image Buffer
-- =============================================================================
--
-- Receives ONE binarized MNIST image.
--
-- The `training.py` script currently converts every pixel to:
--
--     0 when pixel <= 127
--     1 when pixel > 127
--
-- Therefore each incoming pixel can initially be represented by one bit.
--
-- The buffer converts the bit into `data_t` so that the layer MAC can use the
-- same data representation as subsequent activations.
--
-- =============================================================================

entity input_buffer is
    port (
        clk : in std_logic;
        rst : in std_logic;

        pixel_in    : in std_logic;
        pixel_valid : in std_logic;

        clear       : in std_logic;
        image_ready : out std_logic;

        read_address : in natural range 0 to INPUT_SIZE - 1;
        read_data    : out data_t
    );
end entity input_buffer;

architecture rtl of input_buffer is

    -- This would be used if the Python script did not convert each pixel
    -- to either 0 or 1.
    --
    -- type memory_t is array (0 to INPUT_SIZE - 1) of data_t;
    -- signal memory : memory_t := (others => (others => '0'));

    -- Each pixel has already been converted to 0 or 1, so only one bit
    -- is required to store each pixel.
    signal memory : std_logic_vector(INPUT_SIZE - 1 downto 0) := (others => '0');

    signal write_index : natural range 0 to INPUT_SIZE - 1 := 0;
    signal image_ready_i : std_logic := '0';

begin

    image_ready <= image_ready_i;

    -- Combinational read is sufficient for this first implementation.
    --
    -- The image buffer stores each binarized pixel using one bit, but the
    -- neural-network layers operate on data_t values. Convert the selected
    -- pixel to signed 0 or signed 1 when it leaves the input buffer.
    read_data <= to_signed(1, DATA_WIDTH) when memory(read_address) = '1' else to_signed(0, DATA_WIDTH);

    process(clk)
    begin

        if rising_edge(clk) then
            if rst = '1' or clear = '1' then
                write_index   <= 0;
                image_ready_i <= '0';
            elsif pixel_valid = '1' then
                -- Store the already-binarized pixel directly as one bit.
                memory(write_index) <= pixel_in;

                -- The previous implementation stored each pixel as a
                -- DATA_WIDTH-bit signed value:
                --
                -- if pixel_in = '1' then
                --     memory(write_index) <= to_signed(1, DATA_WIDTH);
                -- else
                --     memory(write_index) <= to_signed(0, DATA_WIDTH);
                -- end if;
                --
                -- This is unnecessary because training.py guarantees that
                -- every incoming pixel is already either 0 or 1.

                -- Assert image_ready after the last pixel is received.
                if write_index = INPUT_SIZE - 1 then
                    write_index   <= 0;
                    image_ready_i <= '1';
                else
                    write_index <= write_index + 1;
                end if;
            end if;
        end if;

    end process;

end architecture rtl;