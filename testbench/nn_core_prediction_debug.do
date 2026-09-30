# =============================================================================
# FPGA OCR Neural Network - Prediction Debug
# =============================================================================
#
# Purpose:
#
#   Determine whether the wrong final prediction comes from:
#
#       1. incorrect Layer 4 logits, or
#       2. an incorrect ArgMax implementation.
#
# Current known result:
#
#       expected_digit  = 5
#       predicted_digit = 3
#
# The key comparison is therefore:
#
#       layer4_buffer(0..9)
#
# followed by:
#
#       ArgMax result
#
# =============================================================================

# -----------------------------------------------------------------------------
# Testbench
# -----------------------------------------------------------------------------

add wave -divider {Testbench}

add wave /nn_core_tb/clk
add wave /nn_core_tb/rst

add wave /nn_core_tb/result_valid

add wave -radix unsigned /nn_core_tb/expected_digit
add wave -radix unsigned /nn_core_tb/predicted_digit

add wave -divider {ArgMax Timing}

add wave sim:/nn_core_tb/dut/state
add wave sim:/nn_core_tb/dut/argmax_start
add wave sim:/nn_core_tb/dut/argmax_done

add wave sim:/nn_core_tb/dut/argmax_inst/start
add wave sim:/nn_core_tb/dut/argmax_inst/done

add wave -radix unsigned sim:/nn_core_tb/dut/argmax_inst/index
add wave -radix decimal sim:/nn_core_tb/dut/argmax_inst/max_value
add wave -radix unsigned sim:/nn_core_tb/dut/argmax_inst/max_index
add wave -radix unsigned sim:/nn_core_tb/dut/argmax_inst/class_out

add wave sim:/nn_core_tb/result_valid
add wave -radix unsigned sim:/nn_core_tb/predicted_digit


add wave -divider {Layer 4 Wide Logits}

add wave -radix decimal /nn_core_tb/dut/layer4_buffer(0)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(1)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(2)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(3)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(4)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(5)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(6)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(7)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(8)
add wave -radix decimal /nn_core_tb/dut/layer4_buffer(9)

add wave -divider {Layer 4 Wide Output}

add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/accumulator
add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/shifted_value
add wave -radix decimal /nn_core_tb/dut/dense_layer_inst/output_logit_data
add wave /nn_core_tb/dut/dense_layer_inst/output_write_enable
add wave -radix unsigned /nn_core_tb/dut/dense_layer_inst/output_address

add wave -divider {ArgMax Wide}

add wave /nn_core_tb/dut/argmax_inst/start
add wave /nn_core_tb/dut/argmax_inst/done
add wave -radix unsigned /nn_core_tb/dut/argmax_inst/index
add wave -radix decimal /nn_core_tb/dut/argmax_inst/max_value
add wave -radix unsigned /nn_core_tb/dut/argmax_inst/max_index
add wave -radix unsigned /nn_core_tb/dut/argmax_inst/class_out

# -----------------------------------------------------------------------------
# Wave Window
# -----------------------------------------------------------------------------

configure wave -namecolwidth 350
configure wave -valuecolwidth 150
configure wave -timelineunits us

update
wave zoom full