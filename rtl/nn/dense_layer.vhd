library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

-- =============================================================================
-- Parallel Dense-Layer Engine
-- =============================================================================
--
-- Processes all output neurons in parallel.
--
-- One input activation is consumed per clock.
--
-- For each input activation:
--
--     output 0 accumulator += input * weight(0)
--     output 1 accumulator += input * weight(1)
--     ...
--     output N accumulator += input * weight(N)
--
-- All operations above occur in the same clock cycle.
--
-- The maximum parallelism is 64 output neurons because Layer 1 and Layer 2
-- both contain 64 outputs.
--
-- The same hardware is reused by all four layers.
--
-- =============================================================================

entity dense_layer is

    generic (
        MAX_INPUTS  : positive := LAYER1_INPUTS;
        MAX_OUTPUTS : positive := LAYER1_OUTPUTS
    );

    port (
        -- Clock and reset
        clk : in std_logic;
        rst : in std_logic;
        -- Start signal from nn_core
        start : in std_logic;

        input_count : in natural range 1 to MAX_INPUTS;
        output_count : in natural range 1 to MAX_OUTPUTS;
        apply_relu : in std_logic;
        binary_input_mode : in std_logic;
        -- Current input requested from nn_core.
        input_address : out natural range 0 to MAX_INPUTS - 1;
        input_data : in data_t;

        -- All weights associated with the current input.
        weight_data : in weight_array_t(0 to MAX_OUTPUTS - 1);

        -- All completed layer outputs.
        output_data : out data_array_t(0 to MAX_OUTPUTS - 1);

        -- Wide outputs used by Layer 4 before saturation.
        output_logit_data : out logit_array_t(0 to MAX_OUTPUTS - 1);

        busy : out std_logic;
        done : out std_logic
    );

end entity dense_layer;


architecture rtl of dense_layer is

    type state_t is (
        IDLE,
        ACCUMULATE,
        DONE_STATE);

    signal state : state_t := IDLE;

    signal input_index : natural range 0 to MAX_INPUTS - 1 := 0;

    -- One independent accumulator for every possible output neuron.
    signal accumulators : accumulator_array_t(0 to MAX_OUTPUTS - 1) := (others => (others => '0'));
    signal shifted_values : accumulator_array_t(0 to MAX_OUTPUTS - 1) := (others => (others => '0'));
    signal saturated_values : data_array_t(0 to MAX_OUTPUTS - 1) := (others => (others => '0'));
    signal relu_values : data_array_t(0 to MAX_OUTPUTS - 1) := (others => (others => '0'));

    -- Enables post-processing only for output lanes belonging to the currently
    -- configured layer.
    signal output_lane_enable : std_logic_vector(0 to MAX_OUTPUTS - 1) := (others => '0');

begin
    busy <= '0' when state = IDLE else '1';
    input_address <= input_index;

    -- =========================================================================
    -- Output-Lane Enable
    -- =========================================================================
    --
    -- Only output lanes belonging to the currently configured layer are active.
    -- Unused parallel lanes remain disabled.
    --
    process(all)
    begin
        output_lane_enable <= (others => '0');
        for output_index in 0 to MAX_OUTPUTS - 1 loop
            if output_index < output_count then
                output_lane_enable(output_index) <= '1';
            end if;
        end loop;
    end process;

    -- =========================================================================
    -- Fixed-Point Post-Processing
    -- =========================================================================
    output_processing_gen : for output_index in 0 to MAX_OUTPUTS - 1 generate
        -- Layer 1:
        --
        --     binary x Q1.14 -> Q1.14
        --
        -- Layers 2-4:
        --
        --     Q1.14 x Q1.14 -> Q2.28
        --
        -- and therefore require a 14-bit right shift.
        shifted_values(output_index) <=
            accumulators(output_index)
            when output_lane_enable(output_index) = '1' and
                binary_input_mode = '1'
            else shift_right(
                accumulators(output_index),
                FRACTIONAL_BITS
            )
            when output_lane_enable(output_index) = '1'
            else (others => '0');
                -- Preserve the full-width result for the final layer.
                output_logit_data(output_index) <= shifted_values(output_index);

        saturate_inst : entity work.saturate
            port map (
                enable => output_lane_enable(output_index),
                data_in => shifted_values(output_index),
                data_out => saturated_values(output_index)
            );

        relu_inst : entity work.relu
            port map (
                data_in => saturated_values(output_index),
                data_out => relu_values(output_index)
            );

        output_data(output_index) <= relu_values(output_index) when apply_relu = '1' else saturated_values(output_index);

    end generate output_processing_gen;

    -- =========================================================================
    -- Parallel Dense-Layer Controller
    -- =========================================================================

    process(clk)

        variable product : signed(DATA_WIDTH + WEIGHT_WIDTH - 1 downto 0);
        variable contribution :accumulator_t;

    begin

        if rising_edge(clk) then
            if rst = '1' then
                state <= IDLE;
                input_index <= 0;
                accumulators <= (others => (others => '0'));
                done <= '0';
            else
                done <= '0';
                case state is
                    -- =========================================================
                    -- IDLE
                    -- =========================================================
                    when IDLE =>
                        if start = '1' then
                            input_index <= 0;
                            -- Every output neuron begins with an independent
                            -- zero accumulator.
                            accumulators <= (others => (others => '0'));
                            state <= ACCUMULATE;
                        end if;
                    -- =========================================================
                    -- Parallel Multiply / Accumulate
                    -- =========================================================
                    when ACCUMULATE =>
                        -- All output neurons are processed concurrently.
                        for output_index in 0 to MAX_OUTPUTS - 1 loop
                            if output_index < output_count then

                                assert not is_x(std_logic_vector(weight_data(output_index)))
                                    report "dense_layer: active weight contains U/X"
                                    severity failure;

                                assert not is_x(std_logic_vector(input_data))
                                    report "dense_layer: input_data contains U/X"
                                    severity failure;

                                if binary_input_mode = '1' then
                                    -- Layer 1 multiplier bypass.
                                    if input_data = to_signed(1, DATA_WIDTH) then
                                        contribution := resize(weight_data(output_index),ACCUMULATOR_WIDTH);
                                    else
                                        contribution := (others => '0');
                                    end if;
                                else
                                    -- Layers 2 through 4 use one multiplier per
                                    -- active output neuron.
                                    product := input_data * weight_data(output_index);
                                    contribution := resize(product,ACCUMULATOR_WIDTH);
                                end if;
                                accumulators(output_index) <= accumulators(output_index) + contribution;
                            end if;
                        end loop;
                        -- Every output neuron has consumed the same input.
                        --
                        -- Advance to the next input only once.
                        if input_index = input_count - 1 then
                            state <= DONE_STATE;
                        else
                            input_index <= input_index + 1;
                        end if;
                    -- =========================================================
                    -- Layer Complete
                    -- =========================================================
                    when DONE_STATE =>
                        done <= '1';
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;

end architecture rtl;