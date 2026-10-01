library ieee;
use ieee.std_logic_1164.all;

-- =============================================================================
-- Neural-Network Weight Initialization Package
-- =============================================================================
--
-- Defines the Memory Initialization Files used by Quartus synthesis and Questa
-- RTL simulation.
--
-- Both sets of constants point to the same physical files.
--
-- =============================================================================

package nn_weight_init_pkg is

    -- =========================================================================
    -- Quartus Synthesis Paths
    -- =========================================================================
    --
    -- Quartus project directory:
    --
    --     fpga-ocr/quartus/
    --
    constant LAYER1_WEIGHT_INIT_FILE : string := "../model/quartus_mif/matrix1.mif";
    constant LAYER2_WEIGHT_INIT_FILE : string := "../model/quartus_mif/matrix2.mif";
    constant LAYER3_WEIGHT_INIT_FILE : string := "../model/quartus_mif/matrix3.mif";
    constant LAYER4_WEIGHT_INIT_FILE : string := "../model/quartus_mif/matrix4.mif";

    -- =========================================================================
    -- Questa RTL-Simulation Paths
    -- =========================================================================
    --
    -- NativeLink simulation directory:
    --
    --     fpga-ocr/quartus/simulation/questa/
    --
    constant LAYER1_WEIGHT_SIM_FILE : string := "../../../model/quartus_mif/matrix1.mif";
    constant LAYER2_WEIGHT_SIM_FILE : string := "../../../model/quartus_mif/matrix2.mif";
    constant LAYER3_WEIGHT_SIM_FILE : string := "../../../model/quartus_mif/matrix3.mif";
    constant LAYER4_WEIGHT_SIM_FILE : string := "../../../model/quartus_mif/matrix4.mif";

end package nn_weight_init_pkg;