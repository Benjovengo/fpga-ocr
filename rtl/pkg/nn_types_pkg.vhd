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
    -- Generic array of activation values.
    type data_array_t is array (natural range <>) of data_t;
    -- Generic array of weights.
    type weight_array_t is array (natural range <>) of weight_t;

end package nn_types_pkg;