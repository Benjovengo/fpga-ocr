# =============================================================================
# FPGA OCR Parallel Neural-Network Wave Setup
# =============================================================================
#
# Questa waveform setup for the parallel dense-layer implementation.
#
# This file focuses on:
#
#   - current test image and expected/predicted digit
#   - top-level FSM state
#   - dense-layer control
#   - shared dense input address
#   - layer-specific weight-ROM input indexes
#   - parallel weight vectors
#   - layer output buffers
#   - Layer-4 logits
#   - ArgMax operation
#
# Expected project hierarchy:
#
#   nn_core_tb
#     |
#     +-- dut : nn_core
#           |
#           +-- layer1_rom
#           +-- layer2_rom
#           +-- layer3_rom
#           +-- layer4_rom
#           +-- dense_layer_inst
#           +-- argmax_inst
#
# Run after the design has been loaded with vsim:
#
#   do ../../../testbench/nn_core_parallel_wave.do
#
# Adjust the relative path if Questa was started from another directory.
#
# =============================================================================

# -----------------------------------------------------------------------------
# Clear Existing Wave Window
# -----------------------------------------------------------------------------

delete wave *

# -----------------------------------------------------------------------------
# Testbench / Classification Result
# -----------------------------------------------------------------------------

add wave -divider {Testbench / Classification}

add wave -radix unsigned sim:/nn_core_tb/test_index

# expected_labels is an array containing one expected digit for every image.
# Keep it as one expandable object in the Wave window and use test_index to
# identify the currently active element.
add wave -radix unsigned sim:/nn_core_tb/expected_labels

add wave -radix unsigned sim:/nn_core_tb/predicted_digit
add wave sim:/nn_core_tb/result_valid

add wave -radix unsigned sim:/nn_core_tb/pass_count
add wave -radix unsigned sim:/nn_core_tb/fail_count
add wave -radix unsigned sim:/nn_core_tb/inference_cycles

# -----------------------------------------------------------------------------
# Top-Level Neural-Network Control
# -----------------------------------------------------------------------------

add wave -divider {NN Core FSM}

add wave sim:/nn_core_tb/dut/state
add wave sim:/nn_core_tb/dut/input_ready
add wave sim:/nn_core_tb/dut/dense_start
add wave sim:/nn_core_tb/dut/dense_done
add wave sim:/nn_core_tb/dut/argmax_start
add wave sim:/nn_core_tb/dut/argmax_done

add wave -radix unsigned sim:/nn_core_tb/dut/dense_input_count
add wave -radix unsigned sim:/nn_core_tb/dut/dense_output_count

# -----------------------------------------------------------------------------
# Dense-Layer Input Address
# -----------------------------------------------------------------------------

add wave -divider {Dense Input}

add wave -radix unsigned sim:/nn_core_tb/dut/dense_input_address
add wave -radix decimal sim:/nn_core_tb/dut/dense_input_data

# -----------------------------------------------------------------------------
# Layer-Specific Weight-ROM Input Indexes
# -----------------------------------------------------------------------------
#
# These signals are the important range-protection signals added after the
# parallel ROM change.
#
# Only the ROM belonging to the active layer should follow dense_input_address.
# The other three ROM indexes should remain zero.
#
# In particular:
#
#   Layer 4 legal range = 0 .. 31
#
# so layer4_weight_input_index must never become 32 or greater.
#
# -----------------------------------------------------------------------------

add wave -divider {Weight ROM Input Indexes}

add wave -radix unsigned sim:/nn_core_tb/dut/layer1_weight_input_index
add wave -radix unsigned sim:/nn_core_tb/dut/layer2_weight_input_index
add wave -radix unsigned sim:/nn_core_tb/dut/layer3_weight_input_index
add wave -radix unsigned sim:/nn_core_tb/dut/layer4_weight_input_index

# -----------------------------------------------------------------------------
# Parallel Weight Vectors
# -----------------------------------------------------------------------------
#
# The arrays are added as expandable objects.
#
# Expand the active layer in the Wave window to inspect individual lanes:
#
#   layer1_weight_data(0) ... layer1_weight_data(63)
#   layer2_weight_data(0) ... layer2_weight_data(63)
#   layer3_weight_data(0) ... layer3_weight_data(31)
#   layer4_weight_data(0) ... layer4_weight_data(9)
#
# -----------------------------------------------------------------------------

add wave -divider {Parallel Weight Vectors}

add wave -radix hex sim:/nn_core_tb/dut/layer1_weight_data
add wave -radix hex sim:/nn_core_tb/dut/layer2_weight_data
add wave -radix hex sim:/nn_core_tb/dut/layer3_weight_data
add wave -radix hex sim:/nn_core_tb/dut/layer4_weight_data

add wave -radix hex sim:/nn_core_tb/dut/dense_weight_data

# -----------------------------------------------------------------------------
# Parallel Dense-Layer Internals
# -----------------------------------------------------------------------------

add wave -divider {Dense Layer Internals}

add wave sim:/nn_core_tb/dut/dense_layer_inst/state
add wave -radix unsigned sim:/nn_core_tb/dut/dense_layer_inst/input_index

# Parallel accumulator and post-processing arrays.
# These are expandable in the Wave window.
add wave -radix decimal sim:/nn_core_tb/dut/dense_layer_inst/accumulators
add wave -radix decimal sim:/nn_core_tb/dut/dense_layer_inst/shifted_values
add wave -radix decimal sim:/nn_core_tb/dut/dense_layer_inst/saturated_values
add wave -radix decimal sim:/nn_core_tb/dut/dense_layer_inst/relu_values

# -----------------------------------------------------------------------------
# Layer Output Buffers
# -----------------------------------------------------------------------------

add wave -divider {Layer Output Buffers}

add wave -radix decimal sim:/nn_core_tb/dut/layer1_buffer
add wave -radix decimal sim:/nn_core_tb/dut/layer2_buffer
add wave -radix decimal sim:/nn_core_tb/dut/layer3_buffer

# Layer 4 uses wide logits.
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer

# -----------------------------------------------------------------------------
# Shared Dense Outputs
# -----------------------------------------------------------------------------

add wave -divider {Shared Dense Outputs}

add wave -radix decimal sim:/nn_core_tb/dut/dense_output_data
add wave -radix decimal sim:/nn_core_tb/dut/dense_output_logit_data

# -----------------------------------------------------------------------------
# Layer-4 Logits / ArgMax
# -----------------------------------------------------------------------------

add wave -divider {ArgMax}

add wave sim:/nn_core_tb/dut/argmax_inst/start
add wave sim:/nn_core_tb/dut/argmax_inst/done

add wave -radix unsigned sim:/nn_core_tb/dut/argmax_inst/index
add wave -radix decimal sim:/nn_core_tb/dut/argmax_inst/max_value
add wave -radix unsigned sim:/nn_core_tb/dut/argmax_inst/max_index
add wave -radix unsigned sim:/nn_core_tb/dut/argmax_inst/class_out

# -----------------------------------------------------------------------------
# Useful Individual Layer-4 Logits
# -----------------------------------------------------------------------------
#
# Add all ten explicitly so the final classification can be checked without
# expanding the array.
#
# -----------------------------------------------------------------------------

add wave -divider {Layer 4 Individual Logits}

add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(0)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(1)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(2)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(3)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(4)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(5)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(6)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(7)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(8)
add wave -radix decimal sim:/nn_core_tb/dut/layer4_buffer(9)

# -----------------------------------------------------------------------------
# Wave Window
# -----------------------------------------------------------------------------

configure wave -namecolwidth 360
configure wave -valuecolwidth 180
configure wave -timelineunits us

update
wave zoom full
