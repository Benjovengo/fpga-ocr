library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- =============================================================================
-- FPGA OCR Top-Level Entity
-- =============================================================================
--
-- Top-level entity for the FPGA MNIST OCR implementation.
--
-- This wrapper provides the external interface between the FPGA project and
-- nn_core.
--
-- Neural-network processing remains entirely inside nn_core and its helper
-- blocks.
--
-- Current neural-network topology:
--
--     784 binary pixels
--          |
--          v
--     Layer 1: 784 -> 64
--          |
--         ReLU
--          |
--          v
--     Layer 2: 64 -> 64
--          |
--         ReLU
--          |
--          v
--     Layer 3: 64 -> 32
--          |
--         ReLU
--          |
--          v
--     Layer 4: 32 -> 10
--          |
--          v
--        ArgMax
--          |
--          v
--     predicted_digit
--
-- Only one image is processed at a time.
--
-- pixel_in must already contain the binary representation produced by the
-- Python preprocessing:
--
--     original pixel <= 127 -> '0'
--     original pixel >  127 -> '1'
--
-- The first RTL implementation receives one pixel per pixel_valid assertion.
--
-- =============================================================================

entity ocr_top is
    port (
        -- System clock.
        clk : in std_logic;
        -- Active-low external reset
        rst_n : in std_logic;
        -- Already-binarized input pixel
        pixel_in : in std_logic;
        -- Indicates that pixel_in contains a valid pixel
        pixel_valid : in std_logic;
        -- Indicates that the OCR core can accept input pixels
        input_ready : out std_logic;
        -- Pulses high when predicted_digit contains a valid result
        result_valid : out std_logic;
        -- Predicted MNIST digit from 0 through 9
        predicted_digit : out unsigned(3 downto 0)
    );
end entity ocr_top;

architecture rtl of ocr_top is

    -- nn_core uses an active-high reset
    signal rst : std_logic;

begin

    -- Convert the external active-low reset to the active-high reset used by
    -- the internal neural-network logic
    rst <= not rst_n;

    -- =========================================================================
    -- Neural-Network Core
    -- =========================================================================
    nn_core_inst : entity work.nn_core
        port map (
            clk             => clk,
            rst             => rst,
            pixel_in        => pixel_in,
            pixel_valid     => pixel_valid,
            input_ready     => input_ready,
            result_valid    => result_valid,
            predicted_digit => predicted_digit
        );

end architecture rtl;