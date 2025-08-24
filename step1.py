"""
Step 1: Pre-industrial parameter fitting using piControl data.

This step fits the basic BGC model parameters (Ksoil_0, Kresp_0, Ktfp_0, alpha)
using pre-industrial control data where CO2 and climate are constant.
"""

import os
from step_utils import (
    get_run_output_directory, get_output_filename, run_single_region_model_clean,
    save_fitted_parameters, load_and_filter_data
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
    
    # Build fixed parameters dictionary and parameters to optimize list
    fixed_params = {}
    params_to_optimize = []
    
    # Check main parameters
    if args.Ksoil_0 is not None:
        fixed_params['Ksoil_0'] = args.Ksoil_0
        print(f"Using provided Ksoil_0: {args.Ksoil_0}")
    else:
        params_to_optimize.append('Ksoil_0')
        
    if args.Kresp_0 is not None:
        fixed_params['Kresp_0'] = args.Kresp_0
        print(f"Using provided Kresp_0: {args.Kresp_0}")
    else:
        params_to_optimize.append('Kresp_0')
        
    if args.Ktfp_0 is not None:
        fixed_params['Ktfp_0'] = args.Ktfp_0
        print(f"Using provided Ktfp_0: {args.Ktfp_0}")
    else:
        params_to_optimize.append('Ktfp_0')
        
    if args.alpha is not None:
        fixed_params['alpha'] = args.alpha
        print(f"Using provided alpha: {args.alpha}")
    else:
        params_to_optimize.append('alpha')
    
    # Step 1 doesn't use climate sensitivity parameters - they should be zero
    # Only add them if explicitly provided (for testing purposes)
    if args.Ktfp_tas0 is not None:
        fixed_params['Ktfp_tas0'] = args.Ktfp_tas0
        print(f"Using provided Ktfp_tas0: {args.Ktfp_tas0}")
    if args.Ktfp_tas1 is not None:
        fixed_params['Ktfp_tas1'] = args.Ktfp_tas1
        print(f"Using provided Ktfp_tas1: {args.Ktfp_tas1}")
    if args.Ktfp_pr0 is not None:
        fixed_params['Ktfp_pr0'] = args.Ktfp_pr0
        print(f"Using provided Ktfp_pr0: {args.Ktfp_pr0}")
    if args.Ktfp_pr1 is not None:
        fixed_params['Ktfp_pr1'] = args.Ktfp_pr1
        print(f"Using provided Ktfp_pr1: {args.Ktfp_pr1}")
    
    if params_to_optimize:
        print(f"Running parameter optimization for: {params_to_optimize}")
    else:
        print("Running with user-provided parameters (no optimization)")
    
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
            
            # Run simulation using the new clean approach
            success, param_dict, results_df, optimization_info = run_single_region_model_clean(
                region, model, "step1", fixed_params, params_to_optimize
            )
            
            if success:
                successful_runs += 1
                all_fitted_params.append(param_dict)
                
                # Save individual results
                from step_utils import get_run_output_directory
                output_filename = get_output_filename("simulation_results", region, model, "step1")
                output_filepath = os.path.join(get_run_output_directory(), output_filename)
                results_df.to_csv(output_filepath, index=False)
                print(f"Simulation results saved to {output_filepath}")
                
                print(f"✅ Success: MSE = {param_dict.get('final_mse', 'N/A')}")
                if optimization_info.get('iterations', 0) > 0:
                    print(f"   Optimization: {optimization_info['iterations']} iterations, {optimization_info['function_evaluations']} function evaluations")
            else:
                failed_runs += 1
                print(f"❌ Failed: {optimization_info.get('error', 'Unknown error')}")
    
    # Save all fitted parameters
    if all_fitted_params:
        save_fitted_parameters(all_fitted_params, "step1")
    
    # Print summary
    print(f"\n=== Step 1 Summary ===")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total combinations: {total_combinations}")
    
    return all_fitted_params, successful_runs, failed_runs
