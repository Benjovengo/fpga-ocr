`timescale 1ns/1ps

// =============================================================================
// Neural-Network Core Testbench
// =============================================================================
//
// Mixed-language simulation:
//
//     RTL       : VHDL
//     Testbench : Verilog
//
// The DUT implements the following sequential MNIST neural network:
//
//     784 binary pixels
//          |
//          v
//     Layer 1: 784 -> 64
//          |
//         ReLU
//          |
//          v
//     Layer 2: 64 -> 64
//          |
//         ReLU
//          |
//          v
//     Layer 3: 64 -> 32
//          |
//         ReLU
//          |
//          v
//     Layer 4: 32 -> 10
//          |
//          v
//        ArgMax
//          |
//          v
//     predicted_digit
//
// Only ONE image is processed at a time.
//
// =============================================================================
// Test Data
// =============================================================================
//
// Test vectors are stored under:
//
//     data/
//
// The testbench expects:
//
//     data/input_image.mem
//     data/expected_class.txt
//
// input_image.mem contains exactly 784 already-binarized pixels:
//
//     0
//     0
//     1
//     0
//     ...
//
// expected_class.txt contains the expected MNIST digit:
//
//     7
//
// These files should be generated from data/train.csv by the Python reference
// or vector-generation script.
//
// =============================================================================
// Layer 1 Optimization
// =============================================================================
//
// training.py converts every grayscale MNIST pixel to:
//
//     pixel <= 127 -> 0
//     pixel >  127 -> 1
//
// input_buffer.vhd therefore stores each of the 784 input pixels using only
// one bit.
//
// During Layer 1, nn_core configures:
//
//     dense_binary_input_mode = 1
//
// Therefore dense_layer.vhd replaces:
//
//     input * weight
//
// with:
//
//     input = 0 -> contribution = 0
//     input = 1 -> contribution = weight
//
// Layers 2 through 4 use normal signed fixed-point multiplication.
//
// =============================================================================

module nn_core_tb;

    // =========================================================================
    // Configuration
    // =========================================================================

    // 50 MHz clock.
    localparam CLK_PERIOD = 20;

    // MNIST image dimensions:
    //
    //     28 x 28 = 784 pixels
    localparam INPUT_SIZE = 784;

    // Maximum number of cycles allowed for one complete inference.
    //
    // This prevents an RTL/FSM bug from causing an infinite simulation.
    localparam MAX_INFERENCE_CYCLES = 200000;

    // =========================================================================
    // Test File Paths
    // =========================================================================

    localparam INPUT_IMAGE_FILE =
        "../../../data/input_image.mem";

    localparam EXPECTED_CLASS_FILE =
        "../../../data/expected_class.txt";

    // =========================================================================
    // DUT Inputs
    // =========================================================================

    reg clk;
    reg rst;

    // One already-binarized MNIST pixel.
    reg pixel_in;

    // Indicates that pixel_in contains valid data.
    reg pixel_valid;

    // =========================================================================
    // DUT Outputs
    // =========================================================================

    // High while nn_core is ready to receive image pixels.
    wire input_ready;

    // Pulses high when predicted_digit contains a valid result.
    wire result_valid;

    // Predicted MNIST digit from 0 to 9.
    wire [3:0] predicted_digit;

    // =========================================================================
    // Test Vector Storage
    // =========================================================================

    // One bit is stored for each already-binarized MNIST pixel.
    reg image [0:INPUT_SIZE-1];

    // Expected classification read from expected_class.txt.
    integer expected_digit;

    // General-purpose loop variable.
    integer i;

    // File handle for expected_class.txt.
    integer expected_file;

    // Return value from $fscanf.
    integer scan_result;

    // Number of cycles spent performing inference after the image is received.
    integer inference_cycles;

    //Latency
    time inference_start_time;
    time inference_end_time;
    time inference_latency;
    real inference_latency_ns;
    real inference_latency_us;

    // =========================================================================
    // Device Under Test
    // =========================================================================

    nn_core dut (
        .clk             (clk),
        .rst             (rst),
        .pixel_in        (pixel_in),
        .pixel_valid     (pixel_valid),
        .input_ready     (input_ready),
        .result_valid    (result_valid),
        .predicted_digit (predicted_digit)
    );

    // =========================================================================
    // Clock Generator
    // =========================================================================

    initial begin
        clk = 1'b0;
        forever begin
            #(CLK_PERIOD / 2);
            clk = ~clk;
        end
    end

    // =========================================================================
    // Reset DUT
    // =========================================================================

    task reset_dut;
    begin
        rst = 1'b1;
        pixel_in = 1'b0;
        pixel_valid = 1'b0;

        // Keep reset asserted for five clock cycles.
        repeat (5)
            @(posedge clk);

        rst = 1'b0;

        // Allow nn_core to reach its input-ready state.
        repeat (2)
            @(posedge clk);
    end
    endtask

    // =========================================================================
    // Send One Binary Pixel
    // =========================================================================
    //
    // input_buffer.vhd stores pixel_in on a rising clock edge when pixel_valid
    // is asserted.
    //
    // pixel must already contain either:
    //
    //     0
    //
    // or:
    //
    //     1

    task send_pixel;
        input pixel;
    begin
        pixel_in = pixel;
        pixel_valid = 1'b1;

        @(posedge clk);

        pixel_valid = 1'b0;
    end
    endtask

    // =========================================================================
    // Send One Complete MNIST Image
    // =========================================================================

    task send_image;
    begin

        // Wait until nn_core can accept a new image.
        while (!input_ready)
            @(posedge clk);

        // Send exactly 784 binary pixels.
        for (i = 0; i < INPUT_SIZE; i = i + 1) begin
            send_pixel(image[i]);
        end
    end
    endtask

    // =========================================================================
    // Wait for Classification
    // =========================================================================
    //
    // After the final input pixel is received, inference executes:
    //
    //     Layer 1
    //        |
    //        v
    //     Layer 2
    //        |
    //        v
    //     Layer 3
    //        |
    //        v
    //     Layer 4
    //        |
    //        v
    //      ArgMax
    //
    // Layer 1 uses the binary-input optimization.
    //
    // No second image is accepted during this sequence.

    task wait_for_result;
    begin
        inference_cycles = 0;

        while (!result_valid) begin
            @(posedge clk);

            inference_cycles = inference_cycles + 1;

            if (inference_cycles > MAX_INFERENCE_CYCLES) begin
                $display("");
                $display("ERROR: neural-network inference timeout");
                $display(
                    "Waited %0d cycles without result_valid",
                    inference_cycles
                );
                $fatal;
            end
        end
    end
    endtask

    // =========================================================================
    // Check Classification Result
    // =========================================================================

    task check_result;
    begin
        if (predicted_digit == expected_digit) begin
            $display("");
            $display("========================================");
            $display(" NEURAL NETWORK TEST PASSED");
            $display("========================================");
            $display("Expected digit   : %0d", expected_digit);
            $display("Predicted digit  : %0d", predicted_digit);
            $display("Inference cycles : %0d", inference_cycles);
            $display("");
            $display("Latency          : %0t", inference_latency);
            $display("Latency          : %0.0f ns", inference_latency_ns);
            $display("Latency          : %0.3f us", inference_latency_us);
            $display("");
            $display(
                "Cycle latency     : %0.3f us",
                (inference_cycles * CLK_PERIOD) / 1000.0
            );
            $display("");
        end else begin
            $display("");
            $display("========================================");
            $display(" NEURAL NETWORK TEST FAILED");
            $display("========================================");
            $display("Expected digit   : %0d", expected_digit);
            $display("Predicted digit  : %0d", predicted_digit);
            $display("Inference cycles : %0d", inference_cycles);
            $display("");
            $display("Latency          : %0t", inference_latency);
            $display("Latency          : %0.0f ns", inference_latency_ns);
            $display("Latency          : %0.3f us", inference_latency_us);
            $display("");
            $display(
                "Cycle latency     : %0.3f us",
                (inference_cycles * CLK_PERIOD) / 1000.0
            );
            $display("");
            $fatal;
        end
    end
    endtask

    // =========================================================================
    // Main Test Sequence
    // =========================================================================

    initial begin

        // ---------------------------------------------------------------------
        // Initialize Testbench Signals
        // ---------------------------------------------------------------------

        rst = 1'b0;
        pixel_in = 1'b0;
        pixel_valid = 1'b0;
        inference_cycles = 0;

        // ---------------------------------------------------------------------
        // Load Binary MNIST Image
        // ---------------------------------------------------------------------
        //
        // Source:
        //
        //     data/input_image.mem
        //
        // Expected format:
        //
        //     0
        //     0
        //     1
        //     0
        //     ...
        //
        // The file must contain exactly 784 entries.
        //
        // The vector-generation script must use the same conversion as
        // training.py:
        //
        //     pixel > 127
        //
        // so the RTL and Python implementations receive identical input data.

        $display("Loading input image from:");
        $display("  %s", INPUT_IMAGE_FILE);

        $readmemb(
            INPUT_IMAGE_FILE,
            image
        );

        // ---------------------------------------------------------------------
        // Load Expected Classification
        // ---------------------------------------------------------------------
        //
        // Source:
        //
        //     data/expected_class.txt
        //
        // Example:
        //
        //     7

        $display("Loading expected class from:");
        $display("  %s", EXPECTED_CLASS_FILE);

        expected_file = $fopen(
            EXPECTED_CLASS_FILE,
            "r"
        );

        if (expected_file == 0) begin
            $display("");
            $display(
                "ERROR: unable to open %s",
                EXPECTED_CLASS_FILE
            );
            $fatal;
        end

        scan_result = $fscanf(
            expected_file,
            "%d",
            expected_digit
        );

        if (scan_result != 1) begin
            $display("");
            $display(
                "ERROR: unable to read expected class from %s",
                EXPECTED_CLASS_FILE
            );
            $fatal;
        end

        $fclose(expected_file);

        // ---------------------------------------------------------------------
        // Display Test Configuration
        // ---------------------------------------------------------------------

        $display("");
        $display("========================================");
        $display(" FPGA OCR NEURAL NETWORK TEST");
        $display("========================================");
        $display("Input pixels     : %0d", INPUT_SIZE);
        $display("Expected digit   : %0d", expected_digit);
        $display("Layer 1 mode     : binary optimized");
        $display("Layers 2-4 mode  : fixed-point MAC");
        $display("");

        // ---------------------------------------------------------------------
        // Reset DUT
        // ---------------------------------------------------------------------

        reset_dut();

        // ---------------------------------------------------------------------
        // Send One Image
        // ---------------------------------------------------------------------

        $display(
            "Sending %0d binary MNIST pixels...",
            INPUT_SIZE
        );

        send_image();

        inference_start_time = $time;

        $display(
            "Image received by DUT."
        );

        // ---------------------------------------------------------------------
        // Wait for Sequential Inference
        // ---------------------------------------------------------------------

        $display(
            "Waiting for neural-network inference..."
        );

        wait_for_result();

        inference_end_time = $time;
        inference_latency = inference_end_time - inference_start_time;
        inference_latency_ns = inference_latency;
        inference_latency_us = inference_latency_ns / 1000.0;

        // ---------------------------------------------------------------------
        // Check Classification
        // ---------------------------------------------------------------------
        check_result();

        // ---------------------------------------------------------------------
        // Finish Simulation
        // ---------------------------------------------------------------------

        $display(
            "Simulation completed successfully."
        );

        $finish;
    end

endmodule