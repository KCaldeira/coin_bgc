"""
Step 2: CO2 fertilization effect estimation using SSP585bgc data.

This step estimates the CO2 fertilization parameter (Ktfp_co2) using data where
the biosphere sees CO2 changes but climate is held constant.
"""

import os
from step_utils import (
    get_run_output_directory, get_output_filename, run_single_region_model_clean,
    save_fitted_parameters, load_co2_data
)

def load_step1_parameters(step1_params, region, model):
    """
    Load Step 1 parameters for a specific region/model combination.
    
    Args:
        step1_params: Dictionary of Step 1 results
        region: Geographic region
        model: Climate model
    
    Returns:
        dict: Step 1 parameters for the region/model
    """
    if step1_params and (region, model) in step1_params:
        return step1_params[(region, model)]
    else:
        print(f"Warning: No Step 1 parameters found for {region} / {model}")
        return {}

def run_step2_analysis(args, regions_to_run, models_to_run, step1_params=None, actual_step="step2"):
    """
    Step 2: CO2 fertilization effect estimation using SSP585bgc data.
    
    Args:
        args: Parsed command line arguments
        regions_to_run: List of regions to process
        models_to_run: List of models to process
        step1_params: Dictionary of Step 1 results (optional)
        actual_step: Actual step name for output files
    
    Returns:
        tuple: (all_fitted_params, successful_runs, failed_runs)
    """
    print("=== Step 2: CO2 Fertilization Effect Estimation ===")
    print("Using concatenated historical + SSP585bgc data (CO2 changes, constant climate)")
    
    # Load CO2 data
    co2_df = load_co2_data()
    print(f"Loaded CO2 data: {len(co2_df)} rows")
    
    # Build fixed parameters dictionary and parameters to optimize list
    fixed_params = {}
    params_to_optimize = []
    
    # For Step 2, we should inherit Step 1 parameters and only optimize Ktfp_co2
    # Only add user-provided parameters to fixed_params if explicitly given
    
    # Check main parameters (only if explicitly provided by user)
    if args.Ksoil_0 is not None:
        fixed_params['Ksoil_0'] = args.Ksoil_0
        print(f"Using provided Ksoil_0: {args.Ksoil_0}")
        
    if args.Kresp_0 is not None:
        fixed_params['Kresp_0'] = args.Kresp_0
        print(f"Using provided Kresp_0: {args.Kresp_0}")
        
    if args.Ktfp_0 is not None:
        fixed_params['Ktfp_0'] = args.Ktfp_0
        print(f"Using provided Ktfp_0: {args.Ktfp_0}")
        
    if args.alpha is not None:
        fixed_params['alpha'] = args.alpha
        print(f"Using provided alpha: {args.alpha}")
    
    # Check CO2 parameter
    if args.Ktfp_co2 is not None:
        fixed_params['Ktfp_co2'] = args.Ktfp_co2
        print(f"Using provided Ktfp_co2: {args.Ktfp_co2}")
    else:
        params_to_optimize.append('Ktfp_co2')
    
    # Step 2 doesn't use climate sensitivity parameters - they should be zero
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
            
            # Use Step 1 parameters for this region/model if available
            step_params = fixed_params.copy()
            
            # Add Step 1 parameters
            if step1_params and (region, model) in step1_params:
                step1_param_dict = step1_params[(region, model)]
                step_params.update({
                    'Ksoil_0': step1_param_dict.get('Ksoil_0', step_params.get('Ksoil_0')),
                    'Kresp_0': step1_param_dict.get('Kresp_0', step_params.get('Kresp_0')),
                    'Ktfp_0': step1_param_dict.get('Ktfp_0', step_params.get('Ktfp_0')),
                    'alpha': step1_param_dict.get('alpha', step_params.get('alpha')),
                    'Ktfp_tas0': step1_param_dict.get('Ktfp_tas0', 0.0),
                    'Ktfp_tas1': step1_param_dict.get('Ktfp_tas1', 0.0),
                    'Ktfp_pr0': step1_param_dict.get('Ktfp_pr0', 0.0),
                    'Ktfp_pr1': step1_param_dict.get('Ktfp_pr1', 0.0)
                })
                print(f"Using Step 1 parameters for {region} / {model}")
            
            # For Step 2, we should only optimize Ktfp_co2 if it's not already provided
            region_params_to_optimize = []
            if 'Ktfp_co2' not in step_params:
                region_params_to_optimize.append('Ktfp_co2')
            
            # Run simulation using the new clean approach
            success, param_dict, results_df, optimization_info = run_single_region_model_clean(
                region, model, actual_step, step_params, region_params_to_optimize, co2_df
            )
            
            if success:
                successful_runs += 1
                all_fitted_params.append(param_dict)
                
                # Save individual results
                from step_utils import get_run_output_directory
                output_filename = get_output_filename("simulation_results", region, model, actual_step)
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
        save_fitted_parameters(all_fitted_params, actual_step)
    
    # Print summary
    print(f"\n=== Step 2 Summary ===")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total combinations: {total_combinations}")
    
    return all_fitted_params, successful_runs, failed_runs
