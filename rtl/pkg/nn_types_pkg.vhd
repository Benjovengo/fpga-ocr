library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_config_pkg.all;

-- =============================================================================
-- Neural-Network Type Definitions
-- =============================================================================
--
-- This package contains the reusable signed types and unconstrained arrays
-- required by the neural-network RTL.
--
-- =============================================================================

package nn_types_pkg is

    -- Signed activation value.
    subtype data_t is signed(DATA_WIDTH - 1 downto 0);
    -- Signed quantized weight.
    subtype weight_t is signed(WEIGHT_WIDTH - 1 downto 0);
    -- Wide accumulator used by the MAC datapath.
    subtype accumulator_t is signed(ACCUMULATOR_WIDTH - 1 downto 0);
    -- Final-layer logits remain wide until ArgMax.
    --
    -- Layers 1 through 3 use data_t because their outputs are subsequently used
    -- as fixed-point activations by another dense layer.
    --
    -- Layer 4 does not feed another dense layer, so reducing its result to data_t
    -- would unnecessarily lose information through saturation.
    subtype logit_t is signed(ACCUMULATOR_WIDTH - 1 downto 0);
    
    type logit_array_t is array (natural range <>) of logit_t;
    -- Generic array of activation values.
    type data_array_t is array (natural range <>) of data_t;
    -- Generic array of weights.
    type weight_array_t is array (natural range <>) of weight_t;
    -- Array of wide accumulators used by the parallel dense-layer engine.
    type accumulator_array_t is array (natural range <>) of accumulator_t;

end package nn_types_pkg;