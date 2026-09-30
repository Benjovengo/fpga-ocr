library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;
use work.nn_types_pkg.all;

-- =============================================================================
-- Optimized Sequential Dense-Layer Engine
-- =============================================================================
--
-- Implements one fully connected neural-network layer using a single reusable
-- arithmetic datapath.
--
-- The same instance can execute all four neural-network layers:
--
--     Layer 1: 784 -> 64, ReLU enabled, binary input enabled
--     Layer 2:  64 -> 64, ReLU enabled
--     Layer 3:  64 -> 32, ReLU enabled
--     Layer 4:  32 -> 10, ReLU disabled
--
-- Only one neuron is calculated at a time, and only one input contribution is
-- accumulated per MAC cycle.
--
-- =============================================================================
--
-- General dense-layer operation:
--
--     for each output neuron:
--
--         accumulator = 0
--
--         for each input:
--
--             accumulator += input * weight
--
--         activation = accumulator >> FRACTIONAL_BITS
--
--         activation = saturate(activation)
--
--         if apply_relu:
--             activation = max(0, activation)
--
--         store activation
--
-- =============================================================================
-- Binary Input Optimization
-- =============================================================================
--
-- Layer 1 receives the binarized MNIST image.
--
-- training.py guarantees:
--
--     input = 0 when original pixel <= 127
--     input = 1 when original pixel > 127
--
-- Therefore Layer 1 does not require a general multiplication:
--
--     0 * weight = 0
--     1 * weight = weight
--
-- When binary_input_mode = '1', the multiplier is bypassed and the MAC term is
-- either zero or the weight itself.
--
-- This optimization is only valid for Layer 1.
--
-- Layers 2 through 4 operate on signed fixed-point activations and therefore
-- require normal multiplication.
--
-- =============================================================================

entity dense_layer is

    generic (
        -- Maximum input and output dimensions supported by this shared engine
        -- Layer 1 contains the largest dimensions, therefore these defaults
        -- allow the same instance to execute all four layers
        MAX_INPUTS  : positive := LAYER1_INPUTS;
        MAX_OUTPUTS : positive := LAYER1_OUTPUTS
    );

    port (
        -- Clock and reset signals
        clk : in std_logic;
        rst : in std_logic;
        -- Start one complete dense-layer operation.
        start : in std_logic;

        -- ---------------------------------------------------------------------
        -- Runtime Layer Configuration
        -- ---------------------------------------------------------------------
        -- Number of inputs used by the active layer
        input_count : in natural range 1 to MAX_INPUTS;
        -- Number of output neurons used by the active layer
        output_count : in natural range 1 to MAX_OUTPUTS;

        -- Enable ReLU after the dense operation.
        apply_relu : in std_logic;

        -- Enable the Layer-1 binary-input optimization.
        --      Layer 1:        binary_input_mode = '1'
        --      Layers 2, 3, 4: binary_input_mode = '0'
        binary_input_mode : in std_logic;

        -- ---------------------------------------------------------------------
        -- Input Activation Interface
        -- ---------------------------------------------------------------------
        -- Address of the input value required by the current MAC operation
        input_address : out natural range 0 to MAX_INPUTS - 1;
        -- Current input activation
        input_data : in data_t;

        -- ---------------------------------------------------------------------
        -- Weight-ROM Interface
        -- ---------------------------------------------------------------------
        -- Address of the weight required by the current MAC operation
        weight_address : out natural range 0 to (MAX_INPUTS * MAX_OUTPUTS) - 1;
        -- Weight returned by the selected ROM
        weight_data : in weight_t;

        -- ---------------------------------------------------------------------
        -- Output Activation Interface
        -- ---------------------------------------------------------------------
        -- Pulses high when one completed neuron output must be stored
        output_write_enable : out std_logic;
        -- Destination index for the completed neuron
        output_address : out natural range 0 to MAX_OUTPUTS - 1;
        -- Completed activation value
        output_data : out data_t;
        -- Wide rescaled result before data_t saturation.
        output_logit_data   : out logit_t;

        -- ---------------------------------------------------------------------
        -- Status
        -- ---------------------------------------------------------------------
        -- High while a layer operation is running
        busy : out std_logic;
        -- Pulses high for one clock after the final output neuron is complete
        done : out std_logic
    );

end entity dense_layer;


architecture rtl of dense_layer is

    -- =============================================================================
    -- Finite State Machine (FSM)
    -- =============================================================================
    type state_t is (
        IDLE,
        INIT_NEURON,
        READ_WEIGHT,
        MAC_STATE,
        STORE_OUTPUT,
        DONE_STATE
    );

    -- State initiates as `IDLE`
    signal state : state_t := IDLE;

    -- =============================================================================
    -- Counters
    -- =============================================================================
    -- Input currently being processed
    signal input_index : natural range 0 to MAX_INPUTS - 1 := 0;
    -- Output neuron currently being calculated
    signal output_index : natural range 0 to MAX_OUTPUTS - 1 := 0;

    -- =============================================================================
    -- Arithmetic Signals
    -- =============================================================================
    -- Running sum for the current neuron
    signal accumulator : accumulator_t := (others => '0');
    -- Contribution of the current input/weight pair
    signal mac_term : accumulator_t := (others => '0');
    -- Accumulator value after adding mac_term
    signal accumulator_next : accumulator_t := (others => '0');
    -- Accumulator after fixed-point rescaling
    signal shifted_value : accumulator_t := (others => '0');
    -- Value after saturation to data_t width
    signal saturated_value : data_t := (others => '0');
    -- Value after optional ReLU
    signal relu_value : data_t := (others => '0');
    -- Final value selected for storage
    signal final_output_value : data_t := (others => '0');
    -- Enable saturation only when the current neuron accumulator is complete.
    signal saturate_enable : std_logic := '0';

begin
    -- =============================================================================
    -- Status
    -- =============================================================================
    -- The layer is busy in every state except IDLE
    busy <= '0' when state = IDLE else '1';

    -- =============================================================================
    -- Input Address
    -- =============================================================================
    input_address <= input_index;

    -- =============================================================================
    -- Weight Address Generation
    -- =============================================================================
    weight_address <= (input_index * output_count) + output_index;

    -- =============================================================================
    -- MAC Term Generation
    -- =============================================================================
    --
    -- In binary_input_mode, the multiplier is bypassed:
    --
    --     input = 0 -> mac_term = 0
    --     input = 1 -> mac_term = weight
    --
    -- In normal mode:
    --
    --     mac_term = input * weight

    -- process(binary_input_mode,input_data,weight_data)
    -- The other parameters are constants
    --process(all)
    process(binary_input_mode, input_data, weight_data)
        variable product :
            signed(DATA_WIDTH + WEIGHT_WIDTH - 1 downto 0);
    begin
        if binary_input_mode = '1' then
            if input_data = to_signed(1, DATA_WIDTH) then
                mac_term <= resize(weight_data,ACCUMULATOR_WIDTH);
            else
                mac_term <= (others => '0');
            end if;
        else
            product := input_data * weight_data;

            mac_term <= resize(product,ACCUMULATOR_WIDTH);
        end if;
    end process;

    -- =============================================================================
    -- Accumulator
    -- =============================================================================
    accumulator_next <= accumulator + mac_term;

    -- =============================================================================
    -- Fixed-Point Rescaling
    -- =============================================================================
    --
    -- Layer 1 uses binary inputs rather than Q1.14 activations.
    --
    -- For Layer 1:
    --
    --     binary input x Q1.14 weight -> Q1.14
    --
    -- Therefore the accumulator is already in Q1.14 format and must NOT be shifted.
    --
    -- For Layers 2 through 4:
    --
    --     Q1.14 activation x Q1.14 weight -> Q2.28
    --
    -- Therefore the accumulator must be shifted right by FRACTIONAL_BITS to restore
    -- the Q1.14 activation representation.
    process(binary_input_mode, accumulator)
    begin
        if binary_input_mode = '1' then
            shifted_value <= accumulator;
        else
            shifted_value <= shift_right(accumulator,FRACTIONAL_BITS);
        end if;
    end process;

    -- Preserve the rescaled accumulator at full width for Layer 4.
    --
    -- nn_core uses this output only for the final layer.
    output_logit_data <= shifted_value;

    -- =============================================================================
    -- Saturation
    -- =============================================================================
    --
    -- Saturate the rescaled accumulator to DATA_WIDTH before applying ReLU.
    --
    -- The saturation block is enabled only when the complete neuron result is
    -- available in STORE_OUTPUT.
    saturate_enable <= '1' when state = STORE_OUTPUT else '0';

    saturate_inst : entity work.saturate
        port map (
            enable   => saturate_enable,
            data_in  => shifted_value,
            data_out => saturated_value
        );

    -- =============================================================================
    -- ReLU
    -- =============================================================================

    relu_inst : entity work.relu
        port map (
            data_in  => saturated_value,
            data_out => relu_value
        );

    -- =============================================================================
    -- Final Output Selection
    -- =============================================================================
    --
    -- Layers 1 through 3 use ReLU.
    --
    -- Layer 4 writes the signed output directly because ArgMax must compare the
    -- original output logits.

    final_output_value <= relu_value when apply_relu = '1' else saturated_value;

    -- =============================================================================
    -- Sequential Dense-Layer Controller
    -- =============================================================================
    process(clk)
    begin

        if rising_edge(clk) then

            if rst = '1' then
                -- ----------------------------------------------------------------
                -- Reset
                -- ----------------------------------------------------------------
                state <= IDLE;
                input_index  <= 0;
                output_index <= 0;
                accumulator <= (others => '0');
                output_write_enable <= '0';
                output_address <= 0;
                output_data <= (others => '0');
                done <= '0';
            else
                -- output_write_enable and done are single-cycle pulses
                output_write_enable <= '0';
                done                <= '0';
                case state is
                    -- =========================================================
                    -- IDLE
                    -- =========================================================
                    when IDLE =>
                        -- Wait until nn_core requests a new layer operation.
                        if start = '1' then
                            output_index <= 0;
                            state <= INIT_NEURON;
                        end if;

                    -- =========================================================
                    -- Initialize Current Neuron
                    -- =========================================================
                    when INIT_NEURON =>
                        -- Start with the first input.
                        input_index <= 0;
                        -- Clear the neuron accumulator.
                        accumulator <= (others => '0');
                        -- The selected weight ROM uses synchronous output
                        --
                        -- READ_WEIGHT provides the required latency before the
                        -- MAC consumes weight_data
                        state <= READ_WEIGHT;

                    -- =========================================================
                    -- Wait for Weight ROM
                    -- =========================================================
                    when READ_WEIGHT =>
                        -- input_address and weight_address are already generated
                        -- combinationally from the current counters
                        -- Wait one clock so synchronous ROM data becomes valid
                        state <= MAC_STATE;

                    -- =========================================================
                    -- Multiply / Accumulate
                    -- =========================================================
                    when MAC_STATE =>
                        -- Accumulate the current input contribution.
                        accumulator <= accumulator_next;

                        if input_index = input_count - 1 then
                            -- --------------------------------------------------
                            -- Current neuron complete
                            -- --------------------------------------------------
                            --
                            -- accumulator_next contains the contribution of the
                            -- final input
                            --
                            -- Store it before entering STORE_OUTPUT so the
                            -- combinational fixed-point processing sees the
                            -- complete neuron sum
                            state <= STORE_OUTPUT;
                        else
                            -- --------------------------------------------------
                            -- Process next input
                            -- --------------------------------------------------
                            input_index <= input_index + 1;
                            state <= READ_WEIGHT;
                        end if;

                    -- =========================================================
                    -- Store Neuron Output
                    -- =========================================================
                    when STORE_OUTPUT =>
                        -- The combinational datapath has already performed:
                        --
                        -- accumulator -> shift right -> saturation -> optional ReLU
                        --
                        -- Therefore no dedicated states are required for those
                        -- operations

                        output_write_enable <= '1';
                        output_address <= output_index;
                        output_data <= final_output_value;

                        if output_index = output_count - 1 then
                            -- All neurons in the active layer are complete
                            state <= DONE_STATE;
                        else
                            -- Advance to the next neuron
                            output_index <= output_index + 1;
                            state <= INIT_NEURON;
                        end if;

                    -- =========================================================
                    -- Layer Complete
                    -- =========================================================
                    when DONE_STATE =>
                        -- Pulse done for one clock cycle.
                        done <= '1';
                        state <= IDLE;
                end case;

            end if;

        end if;

    end process;

end architecture rtl;