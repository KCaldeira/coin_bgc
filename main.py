import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse

# Import step modules
from step1 import run_step1_analysis
from step2 import run_step2_analysis, load_step1_parameters
from step3 import run_step3_analysis, load_step_parameters
from step_utils import get_available_regions_and_models

def run_complete_analysis(args, step):
    """
    Top-level function that orchestrates the complete analysis.
    
    Args:
        args: Parsed command line arguments
        step: Which step to run ("step1", "step2", "step3", or "all")
    """
    print(f"=== COIN-BGC Simulation ===")
    print(f"Step: {step}")
    
    # Setup analysis environment
    output_dir = setup_analysis_environment(args)
    
    # Determine which regions and models to run
    regions_to_run, models_to_run = determine_regions_and_models_to_run(args)
    
    if not regions_to_run or not models_to_run:
        print("No valid regions or models to run. Exiting.")
        exit(1)
    
    print(f"Will run for {len(regions_to_run)} regions and {len(models_to_run)} models")
    print(f"Total combinations: {len(regions_to_run) * len(models_to_run)}")
    
    # Run the appropriate step(s)
    if step == "step1":
        all_fitted_params, successful_runs, failed_runs = run_step1_analysis(args, regions_to_run, models_to_run)
    elif step == "step2":
        # Try to load Step 1 parameters if available
        step1_params = {}
        step1_files = [f for f in os.listdir("data/output") if f.startswith("fitted_parameters_all_step1_")]
        if step1_files:
            # Use the most recent Step 1 file
            latest_step1 = sorted(step1_files)[-1]
            step1_filepath = os.path.join("data/output", latest_step1)
            step1_params = load_step1_parameters(step1_filepath)
        
        all_fitted_params, successful_runs, failed_runs = run_step2_analysis(args, regions_to_run, models_to_run, step1_params)
    elif step == "step3":
        # Try to load Step 1 and Step 2 parameters if available
        step1_params = {}
        step2_params = {}
        
        step1_files = [f for f in os.listdir("data/output") if f.startswith("fitted_parameters_all_step1_")]
        step2_files = [f for f in os.listdir("data/output") if f.startswith("fitted_parameters_all_step2_")]
        
        if step1_files:
            latest_step1 = sorted(step1_files)[-1]
            step1_filepath = os.path.join("data/output", latest_step1)
            step1_params = load_step_parameters(step1_filepath)
        
        if step2_files:
            latest_step2 = sorted(step2_files)[-1]
            step2_filepath = os.path.join("data/output", latest_step2)
            step2_params = load_step_parameters(step2_filepath)
        
        all_fitted_params, successful_runs, failed_runs = run_step3_analysis(args, regions_to_run, models_to_run, step1_params, step2_params)
    elif step == "all":
        print("Running all steps sequentially...")
        
        # Step 1
        print("\n" + "="*50)
        all_fitted_params, successful_runs, failed_runs = run_step1_analysis(args, regions_to_run, models_to_run)
        
        # Step 2
        print("\n" + "="*50)
        step1_params = {}
        if all_fitted_params:
            # Convert list of dicts to (region, model) keyed dict
            step1_params = {(p['region'], p['model']): p for p in all_fitted_params}
        
        all_fitted_params, successful_runs, failed_runs = run_step2_analysis(args, regions_to_run, models_to_run, step1_params)
        
        # Step 3
        print("\n" + "="*50)
        step2_params = {}
        if all_fitted_params:
            step2_params = {(p['region'], p['model']): p for p in all_fitted_params}
        
        all_fitted_params, successful_runs, failed_runs = run_step3_analysis(args, regions_to_run, models_to_run, step1_params, step2_params)
    else:
        print(f"Unknown step: {step}")
        return
    
    # Print summary
    total_combinations = len(regions_to_run) * len(models_to_run)
    print(f"\n=== Final Summary ===")
    print(f"Total combinations processed: {total_combinations}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/total_combinations*100:.1f}%")
    print(f"Output directory: {output_dir}")
    
    if failed_runs > 0:
        print(f"Warning: {failed_runs} runs failed. Check the output above for details.")

def setup_analysis_environment(args):
    """
    Setup output directories, validate data files, etc.
    """
    from step_utils import setup_output_directory, reset_run_timestamp
    reset_run_timestamp()  # Reset timestamp for new run
    output_dir = setup_output_directory()
    return output_dir

def determine_regions_and_models_to_run(args):
    """
    Determine which regions and models to run based on command line arguments.
    """
    # Get all available regions and models
    all_regions, all_models = get_available_regions_and_models()
    
    if not all_regions or not all_models:
        print("Error: Could not read regions and models from data file")
        return [], []
    
    # Determine regions to run
    if args.regions is not None:
        # Use specific regions list
        regions_to_run = args.regions
        # Validate that all specified regions exist
        invalid_regions = [r for r in regions_to_run if r not in all_regions]
        if invalid_regions:
            print(f"Warning: Invalid regions specified: {invalid_regions}")
            print(f"Available regions: {all_regions[:10]}... (showing first 10)")
            regions_to_run = [r for r in regions_to_run if r in all_regions]
    elif args.region is not None:
        # Use single region
        if args.region not in all_regions:
            print(f"Warning: Region '{args.region}' not found in data")
            print(f"Available regions: {all_regions[:10]}... (showing first 10)")
            regions_to_run = []
        else:
            regions_to_run = [args.region]
    else:
        # Run all regions
        regions_to_run = all_regions
    
    # Determine models to run
    if args.models is not None:
        # Use specific models list
        models_to_run = args.models
        # Validate that all specified models exist
        invalid_models = [m for m in models_to_run if m not in all_models]
        if invalid_models:
            print(f"Warning: Invalid models specified: {invalid_models}")
            print(f"Available models: {all_models}")
            models_to_run = [m for m in models_to_run if m in all_models]
    elif args.model is not None:
        # Use single model
        if args.model not in all_models:
            print(f"Warning: Model '{args.model}' not found in data")
            print(f"Available models: {all_models}")
            models_to_run = []
        else:
            models_to_run = [args.model]
    else:
        # Run all models
        models_to_run = all_models
    
    return regions_to_run, models_to_run

def parse_command_line_args():
    """
    Parse command line arguments for the BGC simulation.
    """
    parser = argparse.ArgumentParser(
        description='Run COIN-BGC model simulation with customizable parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument('--region', type=str, default=None,
                       help='Geographic region for the simulation (if None, will run for all regions)')
    parser.add_argument('--model', type=str, default=None,
                       help='Climate model to use (if None, will run for all models)')
    parser.add_argument('--regions', type=str, nargs='+', default=None,
                       help='List of specific regions to run (overrides --region)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='List of specific models to run (overrides --model)')

    
    # Model parameters (optional - will use optimization if not provided)
    parser.add_argument('--Ksoil_0', type=float, default=None,
                       help='Base soil respiration rate (if None, will be optimized)')
    parser.add_argument('--Kresp_0', type=float, default=None,
                       help='Base plant respiration fraction (if None, will be optimized)')
    parser.add_argument('--Ktfp_0', type=float, default=None,
                       help='Base total factor productivity (if None, will be optimized)')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Power law exponent for production scaling (if None, will be optimized)')
    parser.add_argument('--Ktfp_co2', type=float, default=None,
                       help='CO2 sensitivity of total factor productivity (Step 2 parameter)')
    
    # Climate sensitivity parameters
    parser.add_argument('--Ksoil_tas', type=float, default=0.0,
                       help='Temperature sensitivity of soil respiration')
    parser.add_argument('--Ksoil_pr', type=float, default=0.0,
                       help='Precipitation sensitivity of soil respiration')
    parser.add_argument('--Kresp_tas', type=float, default=0.0,
                       help='Temperature sensitivity of plant respiration')
    parser.add_argument('--Kresp_pr', type=float, default=0.0,
                       help='Precipitation sensitivity of plant respiration')
    parser.add_argument('--Ktfp_tas', type=float, default=0.0,
                       help='Temperature sensitivity of total factor productivity')
    parser.add_argument('--Ktfp_pr', type=float, default=0.0,
                       help='Precipitation sensitivity of total factor productivity')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='data/output',
                       help='Output directory for results')
    parser.add_argument('--step', type=str, default='all',
                       help='Analysis step to run: step1, step2, step3, or all (default: all)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_command_line_args()
    
    # Run the complete analysis
    run_complete_analysis(args, args.step)