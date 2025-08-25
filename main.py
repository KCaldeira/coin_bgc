import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse

# Import step modules
from step1 import run_step1_analysis
from step2 import run_step2_analysis, load_step1_parameters
from step3 import run_step3_analysis, load_step_parameters
from step4 import run_step4_analysis
from step_utils import get_available_regions_and_models, load_step_parameters_from_file

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
            step1_params = load_step_parameters_from_file(step1_filepath)
        
        if step2_files:
            latest_step2 = sorted(step2_files)[-1]
            step2_filepath = os.path.join("data/output", latest_step2)
            step2_params = load_step_parameters_from_file(step2_filepath)
        
        all_fitted_params, successful_runs, failed_runs = run_step3_analysis(args, regions_to_run, models_to_run, step1_params, step2_params)
    elif step == "step4":
        # Try to load Step 1, Step 2, and Step 3 parameters if available
        step1_params = {}
        step2_params = {}
        step3_params = {}
        
        # Search for Step 1 files in main output directory
        step1_files = [f for f in os.listdir("data/output") if f.startswith("fitted_parameters_all_step1_")]
        
        # Search for Step 2 files in main output directory  
        step2_files = [f for f in os.listdir("data/output") if f.startswith("fitted_parameters_all_step2_")]
        
        # Search for Step 3 files in timestamped subdirectories
        step3_files = []
        for subdir in os.listdir("data/output"):
            subdir_path = os.path.join("data/output", subdir)
            if os.path.isdir(subdir_path) and subdir.startswith("run_"):
                step3_files_in_subdir = [f for f in os.listdir(subdir_path) if f.startswith("fitted_parameters_all_step3_")]
                step3_files.extend([os.path.join(subdir, f) for f in step3_files_in_subdir])
        
        if step1_files:
            latest_step1 = sorted(step1_files)[-1]
            step1_filepath = os.path.join("data/output", latest_step1)
            step1_params = load_step_parameters_from_file(step1_filepath)
        
        if step2_files:
            latest_step2 = sorted(step2_files)[-1]
            step2_filepath = os.path.join("data/output", latest_step2)
            step2_params = load_step_parameters_from_file(step2_filepath)
        
        if step3_files:
            latest_step3 = sorted(step3_files)[-1]
            step3_filepath = os.path.join("data/output", latest_step3)
            step3_params = load_step_parameters_from_file(step3_filepath)
        
        all_fitted_params, successful_runs, failed_runs = run_step4_analysis(args, regions_to_run, models_to_run, step1_params, step2_params, step3_params)
    elif step == "all":
        print("Running all steps sequentially...")
        
        # Step 1
        print("\n" + "="*50)
        step1_results, step1_success, step1_failed = run_step1_analysis(args, regions_to_run, models_to_run)
        
        # Convert Step 1 results to the expected format for Step 2
        step1_params = {}
        if step1_results:
            for param_dict in step1_results:
                region = param_dict['region']
                model = param_dict['model']
                step1_params[(region, model)] = param_dict
        
        # Step 2
        print("\n" + "="*50)
        step2_results, step2_success, step2_failed = run_step2_analysis(args, regions_to_run, models_to_run, step1_params)
        
        # Convert Step 2 results to the expected format for Step 3
        step2_params = {}
        if step2_results:
            for param_dict in step2_results:
                region = param_dict['region']
                model = param_dict['model']
                step2_params[(region, model)] = param_dict
        
        # Step 3
        print("\n" + "="*50)
        step3_results, step3_success, step3_failed = run_step3_analysis(args, regions_to_run, models_to_run, step1_params, step2_params)
        
        # Convert Step 3 results to the expected format for Step 4
        step3_params = {}
        if step3_results:
            for param_dict in step3_results:
                region = param_dict['region']
                model = param_dict['model']
                step3_params[(region, model)] = param_dict
        
        # Step 4
        print("\n" + "="*50)
        step4_results, step4_success, step4_failed = run_step4_analysis(args, regions_to_run, models_to_run, step1_params, step2_params, step3_params)
        
        # Use Step 4 results as final output
        step4_params = step4_results
        
        # Summary
        total_success = step1_success + step2_success + step3_success + step4_success
        total_failed = step1_failed + step2_failed + step3_failed + step4_failed
        total_combinations = len(regions_to_run) * len(models_to_run)
        
        print("\n" + "="*50)
        print("=== Final Summary ===")
        print(f"Total combinations processed: {total_combinations}")
        print(f"Successful runs: {total_success}")
        print(f"Failed runs: {total_failed}")
        print(f"Success rate: {total_success/(total_success+total_failed)*100:.1f}%")
        print(f"Output directory: {output_dir}")
        
        all_fitted_params = step4_params  # Use Step 4 results as final output
        successful_runs = total_success
        failed_runs = total_failed
        
        # PDF books will be created separately using --create-pdf-books command
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
    parser.add_argument('--Ktfp_tas0', type=float, default=None,
                       help='Temperature sensitivity parameter 0 for total factor productivity (if None, will be optimized)')
    parser.add_argument('--Ktfp_tas1', type=float, default=None,
                       help='Temperature sensitivity parameter 1 for total factor productivity (if None, will be optimized)')
    parser.add_argument('--Ktfp_pr0', type=float, default=None,
                       help='Precipitation sensitivity parameter 0 for total factor productivity (if None, will be optimized)')
    parser.add_argument('--Ktfp_pr1', type=float, default=None,
                       help='Precipitation sensitivity parameter 1 for total factor productivity (if None, will be optimized)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='data/output',
                       help='Output directory for results')
    parser.add_argument('--step', type=str, default='all',
                       help='Analysis step to run: step1, step2, step3, step4, or all (default: all)')
    
    # PDF creation option
    parser.add_argument('--create-pdf-books', action='store_true',
                       help='Create PDF books from existing results (does not run analysis)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_command_line_args()
    
    # Check if we're just creating PDF books
    if args.create_pdf_books:
        print("=== Creating PDF Books from Existing Results ===")
        from plotting_utils import create_all_books
        create_all_books()
    else:
        # Run the complete analysis
        run_complete_analysis(args, args.step)
        
        # Create step-specific PDF books
        print("\n" + "="*50)
        print("=== Creating PDF Books ===")
        from plotting_utils import create_step1_book, create_step2_book, create_step4_book, create_step3_vs_step4_bookntetheab 
        from step_utils import get_most_recent_output_directory
        
        output_dir = get_most_recent_output_directory()
        if output_dir is None:
            print("No output directories found. Cannot create PDF books.")
        else:
            if args.step == "step1":
                create_step1_book(output_dir)
            elif args.step == "step2":
                create_step2_book(output_dir)
            elif args.step == "step3":
                # Step 3 and 4 are always run together, so create the comparison book
                create_step3_vs_step4_book(output_dir)
            elif args.step == "step4":
                create_step4_book(output_dir)
            elif args.step == "all":
                # Create all books for complete analysis
                from plotting_utils import create_all_books
                create_all_books()