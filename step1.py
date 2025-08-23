"""
Step 1: Pre-industrial parameter fitting using piControl data.

This step fits the basic BGC model parameters (Ksoil_0, Kresp_0, Ktfp_0, alpha)
using pre-industrial control data where CO2 and climate are constant.
"""

import os
from step_utils import (
    get_run_output_directory, get_output_filename, run_single_region_model,
    save_fitted_parameters
)

def run_step1_analysis(args, regions_to_run, models_to_run):
    """
    Step 1: Pre-industrial parameter fitting using piControl data.
    
    Args:
        args: Parsed command line arguments
        regions_to_run: List of regions to process
        models_to_run: List of models to process
    
    Returns:
        tuple: (all_fitted_params, successful_runs, failed_runs)
    """
    print("=== Step 1: Pre-industrial Parameter Fitting ===")
    print("Using piControl data (constant CO2 and climate)")
    
    # Check if user provided specific parameters
    user_params = {}
    has_user_params = False
    
    if args.Ksoil_0 is not None:
        user_params['Ksoil_0'] = args.Ksoil_0
        print(f"Using provided Ksoil_0: {args.Ksoil_0}")
        has_user_params = True
    if args.Kresp_0 is not None:
        user_params['Kresp_0'] = args.Kresp_0
        print(f"Using provided Kresp_0: {args.Kresp_0}")
        has_user_params = True
    if args.Ktfp_0 is not None:
        user_params['Ktfp_0'] = args.Ktfp_0
        print(f"Using provided Ktfp_0: {args.Ktfp_0}")
        has_user_params = True
    if args.alpha is not None:
        user_params['alpha'] = args.alpha
        print(f"Using provided alpha: {args.alpha}")
        has_user_params = True
    
    # Step 1 doesn't use climate sensitivity parameters - they should be zero
    # Only add them if explicitly provided (for testing purposes)
    if args.Ksoil_tas is not None:
        user_params['Ksoil_tas'] = args.Ksoil_tas
        print(f"Using provided Ksoil_tas: {args.Ksoil_tas}")
        has_user_params = True
    if args.Ksoil_pr is not None:
        user_params['Ksoil_pr'] = args.Ksoil_pr
        print(f"Using provided Ksoil_pr: {args.Ksoil_pr}")
        has_user_params = True
    if args.Kresp_tas is not None:
        user_params['Kresp_tas'] = args.Kresp_tas
        print(f"Using provided Kresp_tas: {args.Kresp_tas}")
        has_user_params = True
    if args.Kresp_pr is not None:
        user_params['Kresp_pr'] = args.Kresp_pr
        print(f"Using provided Kresp_pr: {args.Kresp_pr}")
        has_user_params = True
    if args.Ktfp_tas is not None:
        user_params['Ktfp_tas'] = args.Ktfp_tas
        print(f"Using provided Ktfp_tas: {args.Ktfp_tas}")
        has_user_params = True
    if args.Ktfp_pr is not None:
        user_params['Ktfp_pr'] = args.Ktfp_pr
        print(f"Using provided Ktfp_pr: {args.Ktfp_pr}")
        has_user_params = True
    
    if has_user_params:
        print("Running with user-provided parameters (no optimization)")
    else:
        print("Running parameter optimization")
    
    # Store all successful results
    all_fitted_params = []
    successful_runs = 0
    failed_runs = 0
    
    # Run for each region/model combination
    total_combinations = len(regions_to_run) * len(models_to_run)
    current_combination = 0
    
    for region in regions_to_run:
        for model in models_to_run:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Processing {region} / {model}")
            
            # Run simulation for this region/model (no CO2 data for step1)
            success, params_dict, results_df = run_single_region_model(region, model, args, user_params, co2_df=None)
            
            if success:
                successful_runs += 1
                all_fitted_params.append(params_dict)
                
                # Save individual simulation results
                results_filename = get_output_filename("simulation_results", region, model, "step1")
                results_filepath = os.path.join(get_run_output_directory(), results_filename)
                results_df.to_csv(results_filepath, index=False)
                print(f"Simulation results saved to {results_filepath}")
            else:
                failed_runs += 1
                print(f"Failed to process {region} / {model}")
    
    # Save all fitted parameters to a single file
    if all_fitted_params:
        save_fitted_parameters(all_fitted_params, "step1", single_file=True)
    
    print(f"\n=== Step 1 Summary ===")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/total_combinations*100:.1f}%")
    
    return all_fitted_params, successful_runs, failed_runs
