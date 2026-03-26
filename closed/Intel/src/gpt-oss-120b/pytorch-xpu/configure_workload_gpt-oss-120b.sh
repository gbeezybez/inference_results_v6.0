#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Source common functions
source "$(dirname "${BASH_SOURCE[0]}")/mlperf_common_functions.sh"

# Workload-specific parameters
export WORKLOAD="llama2-70b-99.9"
export MODEL="llama2-70b"

# Configure common workload settings
configure_workload_base

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  # Use common base function with llama2-70b specific parameters
  # Primary scenario: Server, Secondary scenario: offline
  workload_specific_run_base "llama2-70b" "server" "Server" "offline"
}
