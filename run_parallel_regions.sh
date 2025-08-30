#!/bin/bash

# COIN-BGC Parallel Execution Script
# 
# This script runs multiple instances of COIN-BGC in parallel using different region patterns
# to efficiently utilize multi-core machines. After all jobs complete, it automatically
# merges the results using concatenate_results.py
#
# Usage:
#   ./run_parallel_regions.sh [num_cores] [workflow_json]
#
# Examples:
#   ./run_parallel_regions.sh 8                                 # Use 8 cores, default workflow  
#   ./run_parallel_regions.sh 12 workflow_schema_example.json  # Use 12 cores, custom workflow
#
# The script divides regions alphabetically to balance the load across cores.

set -e  # Exit on any error

# Default parameters
DEFAULT_CORES=8
DEFAULT_WORKFLOW="workflow_schema_example.json"
DEFAULT_ALPHA="0.5"
DEFAULT_KSOIL="0.025"
DEFAULT_MODELS="ACCESS-ESM1-5"

# Parse command line arguments
NUM_CORES=${1:-$DEFAULT_CORES}
WORKFLOW_JSON=${2:-$DEFAULT_WORKFLOW}
ALPHA=${3:-$DEFAULT_ALPHA}
KSOIL=${4:-$DEFAULT_KSOIL}
MODELS=${5:-$DEFAULT_MODELS}

echo "=== COIN-BGC Parallel Execution Setup ==="
echo "Cores: $NUM_CORES"
echo "Workflow: $WORKFLOW_JSON"
echo "Alpha: $ALPHA"
echo "Ksoil_0: $KSOIL"  
echo "Models: $MODELS"
echo ""

# Verify workflow file exists
if [ ! -f "$WORKFLOW_JSON" ]; then
    echo "❌ Error: Workflow file '$WORKFLOW_JSON' not found"
    exit 1
fi

# Verify virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected. Attempting to activate .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "✅ Activated virtual environment"
    else
        echo "❌ Error: No virtual environment found. Please run 'python -m venv .venv' and 'source .venv/bin/activate'"
        exit 1
    fi
fi

# Create region patterns for parallel execution
# Divide regions alphabetically to balance load
declare -a REGION_PATTERNS=(
    "[A-C]*"    # Afghanistan, Algeria, Angola, Argentina, Armenia, Australia, Austria, Azerbaijan, Bangladesh, Belarus, Belgium, Benin, Bhutan, Bolivia, Bosnia, Botswana, Brazil, Bulgaria, Burkina, Burundi, Cambodia, Cameroon, Canada, Central, Chad, Chile, China, Colombia, Congo, Costa Rica, Croatia, Cuba, Czechia, Côte
    "[D-G]*"    # Dem. Rep. Congo, Denmark, Djibouti, Dominican, Ecuador, Egypt, El Salvador, Eq. Guinea, Eritrea, Estonia, Ethiopia, Fiji, Finland, France, Gabon, Gambia, Georgia, Germany, Ghana, Greece, Greenland, Guatemala, Guinea, Guinea-Bissau, Guyana
    "[H-L]*"    # Honduras, Hungary, Iceland, India, Indonesia, Iran, Iraq, Ireland, Italy, Japan, Jordan, Kazakhstan, Kenya, Kosovo, Kyrgyzstan, Laos, Latvia, Lebanon, Lesotho, Liberia, Libya, Lithuania
    "[M-P]*"    # Madagascar, Malawi, Malaysia, Mali, Mauritania, Mexico, Moldova, Mongolia, Montenegro, Morocco, Mozambique, Myanmar, Namibia, Nepal, Netherlands, New Zealand, Nicaragua, Niger, Nigeria, North Korea, North Macedonia, Norway, Pakistan, Panama, Papua New Guinea, Paraguay, Peru, Philippines, Poland, Portugal
    "[Q-S]*"    # Romania, Russia, S. Sudan, Samoa, Saudi Arabia, Senegal, Serbia, Sierra Leone, Slovakia, Slovenia, Somalia, Somaliland, South Africa, South Korea, Spain, Sri Lanka, Sudan, Suriname, Sweden, Switzerland, Syria
    "[T-Z]*"    # Taiwan, Tajikistan, Tanzania, Thailand, Timor-Leste, Tunisia, Turkey, Turkmenistan, Uganda, Ukraine, United Kingdom, United States of America, Uruguay, Uzbekistan, Venezuela, Vietnam, Yemen, Zambia, Zimbabwe, eSwatini
)

# Adjust patterns based on number of cores requested
if [ $NUM_CORES -le 6 ]; then
    # Use the predefined patterns
    PATTERNS=("${REGION_PATTERNS[@]:0:$NUM_CORES}")
elif [ $NUM_CORES -le 12 ]; then
    # Add more granular patterns for higher core counts
    PATTERNS=(
        "A*" "B*" "[C-D]*" "[E-G]*" "[H-I]*" "[J-L]*" 
        "[M-N]*" "[O-P]*" "[Q-R]*" "S*" "[T-V]*" "[W-Z]*"
    )
    PATTERNS=("${PATTERNS[@]:0:$NUM_CORES}")
else
    # For very high core counts, use single letters where possible
    PATTERNS=(
        "A*" "B*" "C*" "D*" "E*" "F*" "G*" "H*" "I*" 
        "[J-K]*" "L*" "M*" "N*" "[O-P]*" "[Q-R]*" "S*" 
        "T*" "[U-V]*" "[W-Z]*"
    )
    # Take only what we need
    PATTERNS=("${PATTERNS[@]:0:$NUM_CORES}")
fi

echo "🚀 Starting $NUM_CORES parallel jobs with region patterns:"
for i in "${!PATTERNS[@]}"; do
    echo "   Job $((i+1)): ${PATTERNS[i]}"
done
echo ""

# Array to store background process IDs
PIDS=()

# Launch parallel jobs
for i in "${!PATTERNS[@]}"; do
    PATTERN="${PATTERNS[i]}"
    JOB_NUM=$((i+1))
    
    echo "🔄 Starting Job $JOB_NUM: pattern '$PATTERN'"
    
    # Run in background and capture PID
    python main.py \
        --alpha="$ALPHA" \
        --Ksoil_0="$KSOIL" \
        --region-pattern="$PATTERN" \
        --models="$MODELS" \
        --json="$WORKFLOW_JSON" \
        &
    
    PID=$!
    PIDS+=($PID)
    
    echo "   ✅ Job $JOB_NUM (PID $PID) started with pattern: $PATTERN"
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo ""
echo "⏳ All $NUM_CORES jobs launched. Waiting for completion..."
echo "   PIDs: ${PIDS[*]}"

# Function to check if a process is still running
is_running() {
    kill -0 "$1" 2>/dev/null
}

# Monitor jobs
COMPLETED_JOBS=0
FAILED_JOBS=0
START_TIME=$(date +%s)

while [ $COMPLETED_JOBS -lt ${#PIDS[@]} ]; do
    for i in "${!PIDS[@]}"; do
        PID="${PIDS[i]}"
        PATTERN="${PATTERNS[i]}"
        JOB_NUM=$((i+1))
        
        # Skip if this job was already processed
        if [ "$PID" = "DONE" ] || [ "$PID" = "FAILED" ]; then
            continue
        fi
        
        # Check if process is still running
        if ! is_running "$PID"; then
            # Process finished, check exit code
            wait "$PID"
            EXIT_CODE=$?
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ Job $JOB_NUM completed successfully (pattern: $PATTERN)"
                PIDS[i]="DONE"
                COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
            else
                echo "❌ Job $JOB_NUM failed with exit code $EXIT_CODE (pattern: $PATTERN)"
                PIDS[i]="FAILED"
                COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
                FAILED_JOBS=$((FAILED_JOBS + 1))
            fi
        fi
    done
    
    # Brief pause before checking again
    sleep 5
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
SUCCESSFUL_JOBS=$((NUM_CORES - FAILED_JOBS))

echo ""
echo "=============================================="
echo "📊 PARALLEL EXECUTION SUMMARY"
echo "=============================================="
echo "🕒 Total execution time: ${TOTAL_TIME}s"
echo "✅ Successful jobs: $SUCCESSFUL_JOBS/$NUM_CORES"
if [ $FAILED_JOBS -gt 0 ]; then
    echo "❌ Failed jobs: $FAILED_JOBS/$NUM_CORES"
fi
echo ""

# If we have successful jobs, concatenate results
if [ $SUCCESSFUL_JOBS -gt 0 ]; then
    echo "🔗 Concatenating results from successful jobs..."
    
    # Get timestamp pattern for today's runs
    TODAY_PATTERN="run_$(date +%Y%m%d)_*"
    
    python concatenate_results.py "$TODAY_PATTERN"
    
    echo ""
    echo "✅ Parallel execution and concatenation completed!"
    echo "📁 Check data/output/ for individual run results"
    echo "📁 Check data/output/merged_* for concatenated results"
else
    echo "❌ All jobs failed. No results to concatenate."
    exit 1
fi

echo ""
echo "🎉 COIN-BGC parallel execution completed successfully!"