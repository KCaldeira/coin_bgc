"""
Step 3: Climate sensitivity estimation using SSP585 data.

This step fits climate sensitivity parameters (Ksoil_tas, Ksoil_pr, Kresp_tas, Kresp_pr, Ktfp_tas, Ktfp_pr)
using SSP585 data where both CO2 and climate change. Uses parameters from Steps 1 and 2 as starting values.
"""

import os
import pandas as pd
from step_utils import (
    setup_output_directory, get_output_filename, run_single_region_model,
    save_fitted_parameters, load_co2_data
)

def run_step3_analysis(args, regions_to_run, models_to_run, step1_params=None, step2_params=None):
    """
    Step 3: Climate sensitivity estimation using SSP585 data.
    
    Args:
        args: Parsed command line arguments
        regions_to_run: List of regions to process
        models_to_run: List of models to process
        step1_params: Optional parameters from Step 1 (dict with region/model as keys)
        step2_params: Optional parameters from Step 2 (dict with region/model as keys)
    
    Returns:
        tuple: (all_fitted_params, successful_runs, failed_runs)
    """
    print("=== Step 3: Climate Sensitivity Estimation ===")
    print("Using SSP585 data (both CO2 and climate change)")
    
    # Load CO2 data
    print("Loading CO2 concentration data...")
    co2_df = load_co2_data()
    if co2_df is not None:
        print(f"Loaded CO2 data from 1850-{co2_df['year'].max()}")
        print(f"CO2 range: {co2_df['co2'].min():.1f} - {co2_df['co2'].max():.1f} ppm")
    else:
        print("Warning: Could not load CO2 data, using constant pre-industrial value")
    
    # Check if user provided specific parameters
    user_params = {}
    has_user_params = False
    
    # Add Step 1 and Step 2 parameters if provided
    if step1_params:
        print("Using Step 1 parameters as starting values")
    if step2_params:
        print("Using Step 2 parameters as starting values")
    
    # Check for user-provided parameters
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
    if args.Ktfp_co2 is not None:
        user_params['Ktfp_co2'] = args.Ktfp_co2
        print(f"Using provided Ktfp_co2: {args.Ktfp_co2}")
        has_user_params = True
    
    # Add climate sensitivity parameters (these have defaults, so always add them)
    user_params.update({
        'Ksoil_tas': args.Ksoil_tas,
        'Ksoil_pr': args.Ksoil_pr,
        'Kresp_tas': args.Kresp_tas,
        'Kresp_pr': args.Kresp_pr,
        'Ktfp_tas': args.Ktfp_tas,
        'Ktfp_pr': args.Ktfp_pr
    })
    
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
            
            # Use Step 1 and Step 2 parameters for this region/model if available
            step_params = user_params.copy()
            
            # Add Step 1 parameters
            if step1_params and (region, model) in step1_params:
                step1_param_dict = step1_params[(region, model)]
                step_params.update({
                    'Ksoil_0': step1_param_dict.get('Ksoil_0', step_params.get('Ksoil_0')),
                    'Kresp_0': step1_param_dict.get('Kresp_0', step_params.get('Kresp_0')),
                    'Ktfp_0': step1_param_dict.get('Ktfp_0', step_params.get('Ktfp_0')),
                    'alpha': step1_param_dict.get('alpha', step_params.get('alpha')),
                    'Ksoil_tas': step1_param_dict.get('Ksoil_tas', step_params.get('Ksoil_tas')),
                    'Ksoil_pr': step1_param_dict.get('Ksoil_pr', step_params.get('Ksoil_pr')),
                    'Kresp_tas': step1_param_dict.get('Kresp_tas', step_params.get('Kresp_tas')),
                    'Kresp_pr': step1_param_dict.get('Kresp_pr', step_params.get('Kresp_pr')),
                    'Ktfp_tas': step1_param_dict.get('Ktfp_tas', step_params.get('Ktfp_tas')),
                    'Ktfp_pr': step1_param_dict.get('Ktfp_pr', step_params.get('Ktfp_pr'))
                })
                print(f"Using Step 1 parameters for {region} / {model}")
            
            # Add Step 2 parameters
            if step2_params and (region, model) in step2_params:
                step2_param_dict = step2_params[(region, model)]
                step_params.update({
                    'Ktfp_co2': step2_param_dict.get('Ktfp_co2', step_params.get('Ktfp_co2'))
                })
                print(f"Using Step 2 parameters for {region} / {model}")
            
            # Run simulation for this region/model with CO2 data
            success, params_dict, results_df = run_single_region_model(region, model, args, step_params, co2_df=co2_df)
            
            if success:
                successful_runs += 1
                all_fitted_params.append(params_dict)
                
                # Save individual simulation results
                results_filename = get_output_filename("simulation_results", region, model, args.step)
                results_filepath = os.path.join(setup_output_directory(), results_filename)
                results_df.to_csv(results_filepath, index=False)
                print(f"Simulation results saved to {results_filepath}")
                
                # Print climate sensitivity parameters if optimized
                if 'optimization_success' in params_dict:
                    print(f"Climate sensitivity parameters:")
                    print(f"  Ksoil_tas: {params_dict.get('Ksoil_tas', 0):.4f}")
                    print(f"  Ksoil_pr:  {params_dict.get('Ksoil_pr', 0):.4f}")
                    print(f"  Kresp_tas: {params_dict.get('Kresp_tas', 0):.4f}")
                    print(f"  Kresp_pr:  {params_dict.get('Kresp_pr', 0):.4f}")
                    print(f"  Ktfp_tas:  {params_dict.get('Ktfp_tas', 0):.4f}")
                    print(f"  Ktfp_pr:   {params_dict.get('Ktfp_pr', 0):.4f}")
            else:
                failed_runs += 1
                print(f"Failed to process {region} / {model}")
    
    # Save all fitted parameters to a single file
    if all_fitted_params:
        save_fitted_parameters(all_fitted_params, args.step, single_file=True)
    
    print(f"\n=== Step 3 Summary ===")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/total_combinations*100:.1f}%")
    
    # Print climate sensitivity statistics
    if all_fitted_params:
        print(f"Climate sensitivity statistics:")
        climate_params = ['Ksoil_tas', 'Ksoil_pr', 'Kresp_tas', 'Kresp_pr', 'Ktfp_tas', 'Ktfp_pr']
        for param in climate_params:
            values = [p.get(param, 0) for p in all_fitted_params if param in p]
            if values:
                print(f"  {param}:")
                print(f"    Mean: {pd.Series(values).mean():.4f}")
                print(f"    Std:  {pd.Series(values).std():.4f}")
                print(f"    Min:  {min(values):.4f}")
                print(f"    Max:  {max(values):.4f}")
    
    return all_fitted_params, successful_runs, failed_runs

def load_step_parameters(step_output_file):
    """
    Load parameters from a step output file.
    
    Args:
        step_output_file: Path to step fitted parameters file
    
    Returns:
        dict: Dictionary with (region, model) as keys and parameter dict as values
    """
    try:
        if not os.path.exists(step_output_file):
            print(f"Step output file not found: {step_output_file}")
            return {}
        
        step_df = pd.read_csv(step_output_file)
        step_params = {}
        
        for _, row in step_df.iterrows():
            region = row['region']
            model = row['model']
            step_params[(region, model)] = row.to_dict()
        
        print(f"Loaded step parameters for {len(step_params)} region/model combinations")
        return step_params
        
    except Exception as e:
        print(f"Error loading step parameters: {e}")
        return {}
