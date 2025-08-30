#!/bin/bash

# Arrays of parameter values
ALPHAS=(0.25 0.5 1.0)
KSOILS=(0.0125 0.025 0.05)

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel runs..."

# Loop through all combinations
for alpha in "${ALPHAS[@]}"; do
    for ksoil in "${KSOILS[@]}"; do
        # Create a unique log file name
        log_file="logs/run_alpha_${alpha}_ksoil_${ksoil}.log"
        
        echo "Starting: alpha=${alpha}, ksoil=${ksoil}"
        
        # Run in background and redirect output to log file
        (python main.py --alpha="${alpha}" --Ksoil_0="${ksoil}"  --regions "China,Canada,Brazil,Zimbabwe" --json workflow_schema_example.json) > "${log_file}" 2>&1 &
    done
done

# Wait for all background jobs to complete
wait

echo "All runs completed! Check logs/ directory for individual run outputs."
