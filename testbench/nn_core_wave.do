# =============================================================================
# FPGA OCR Neural Network - Questa Debug Waveform
# =============================================================================
#
# Focus:
#   1. Verify Layer 1 ROM selection.
#   2. Verify LAYER_ID = 1.
#   3. Verify package weight reaches selected_weight.
#   4. Verify synchronous ROM output.
#   5. Verify the weight reaches dense_layer.
#   6. Verify the binary Layer 1 MAC.
#   7. Verify accumulation and neuron output.
#
# Known reference:
#
#   LAYER1_WEIGHTS(13056) = x"1826"
#
# Therefore, when:
#
#   LAYER_ID = 1
#   address  = 13056
#
# selected_weight must be x"1826".
# =============================================================================

# -----------------------------------------------------------------------------
# Testbench
# -----------------------------------------------------------------------------

add wave -divider {Testbench}

add wave /nn_core_tb/clk
add wave /nn_core_tb/rst
add wave /nn_core_tb/pixel_in
add wave /nn_core_tb/pixel_valid
add wave /nn_core_tb/input_ready
add wave /nn_core_tb/result_valid

add wave -radix unsigned /nn_core_tb/predicted_digit
add wave -radix unsigned /nn_core_tb/expected_digit

# -----------------------------------------------------------------------------
# Layer 1 ROM
# -----------------------------------------------------------------------------

add wave -divider {Layer 1 ROM}

# LAYER_ID must be 1 for this ROM instance.
add wave -radix unsigned /nn_core_tb/dut/layer1_rom/LAYER_ID

# Address requested by nn_core.
add wave -radix unsigned /nn_core_tb/dut/layer1_rom/address

# Direct value selected from LAYER1_WEIGHTS(address).
#
# At address 13056 this must be:
#
#   x"1826"
#
add wave -radix hex /nn_core_tb/dut/layer1_rom/selected_weight

# Registered ROM value.
# This follows selected_weight by one rising clock edge.
add wave -radix hex /nn_core_tb/dut/layer1_rom/data_out_i

# Entity output of weight_rom.
add wave -radix hex /nn_core_tb/dut/layer1_rom/data_out

# -----------------------------------------------------------------------------
# nn_core Layer 1 Weight Interface
# -----------------------------------------------------------------------------

add wave -divider {Layer 1 Weight Interface}

add wave -radix unsigned /nn_core_tb/dut/layer1_weight_address
add wave -radix hex /nn_core_tb/dut/layer1_weight_data

# -----------------------------------------------------------------------------
# Dense Layer Control
# -----------------------------------------------------------------------------

add wave -divider {Dense Layer Control}

add wave /nn_core_tb/dut/dense_layer_inst/state
add wave /nn_core_tb/dut/dense_layer_inst/binary_input_mode

add wave -radix unsigned /nn_core_tb/dut/dense_layer_inst/input_index
add wave -radix unsigned /nn_core_tb/dut/dense_layer_inst/output_index

add wave -radix unsigned /nn_core_tb/dut/dense_layer_inst/input_address
add wave -radix unsigned /nn_core_tb/dut/dense_layer_inst/weight_address

# -----------------------------------------------------------------------------
# Dense Layer Input and Weight
# -----------------------------------------------------------------------------

add wave -divider {Dense Layer Input and Weight}

add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/input_data
add wave -radix hex /nn_core_tb/dut/dense_layer_inst/weight_data

# Also observe the nn_core weight mux.
add wave -radix unsigned /nn_core_tb/dut/dense_weight_address
add wave -radix hex /nn_core_tb/dut/dense_weight_data

# -----------------------------------------------------------------------------
# Dense Layer MAC
# -----------------------------------------------------------------------------

add wave -divider {Dense Layer MAC}

add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/mac_term
add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/accumulator
add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/accumulator_next

# -----------------------------------------------------------------------------
# Dense Layer Post-processing
# -----------------------------------------------------------------------------

add wave -divider {Dense Layer Post-processing}

add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/shifted_value
add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/saturated_value
add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/relu_value

# -----------------------------------------------------------------------------
# Dense Layer Output
# -----------------------------------------------------------------------------

add wave -divider {Dense Layer Output}

add wave /nn_core_tb/dut/dense_layer_inst/output_write_enable
add wave -radix unsigned /nn_core_tb/dut/dense_layer_inst/output_address
add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/output_data

# -----------------------------------------------------------------------------
# Wave Window
# -----------------------------------------------------------------------------

configure wave -namecolwidth 350
configure wave -valuecolwidth 120
configure wave -timelineunits ns

update