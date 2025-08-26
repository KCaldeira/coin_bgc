"""
Step 1: Pre-industrial parameter fitting using piControl data with steady-state approach.

This step uses steady-state analysis to calculate Ktfp_0 analytically,
then optimizes alpha using the time-dependent piControl simulation.
"""

import os
from step_utils import (
    get_run_output_directory, get_output_filename, run_single_region_model_clean,
    save_fitted_parameters, load_and_filter_data, analyze_steady_state_data,
    calculate_optimal_alpha_step1, run_bgc_simulation
)

def run_step1_analysis(args, regions_to_run, models_to_run):
    """
    Step 1: Pre-industrial parameter fitting using steady-state approach.
    
    Args:
        args: Parsed command line arguments (Ksoil_0 is required)
        regions_to_run: List of regions to process
        models_to_run: List of models to process
    
    Returns:
        tuple: (all_fitted_params, successful_runs, failed_runs)
    """
    print("=== Step 1: Pre-industrial Parameter Fitting (Steady-State Approach) ===")
    print("Using piControl data with steady-state analysis")
    print(f"User-provided Ksoil_0: {args.Ksoil_0}")
    
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
            
            try:
                # Load piControl data for steady-state analysis
                data_df = load_and_filter_data("data/input/Data_regression_piControl.csv", region, model)
                if data_df.empty:
                    print(f"❌ Failed: No data found for {region}/{model}")
                    failed_runs += 1
                    continue
                
                # Perform steady-state analysis
                steady_state_results = analyze_steady_state_data(data_df)
                Kresp_0 = steady_state_results['Kresp_0']
                gpp_mean = steady_state_results['gpp_mean']
                
                # Calculate Cland_0 from steady-state
                # At steady-state: NPP = Ksoil_0 * Cland_0
                # So: Cland_0 = NPP / Ksoil_0
                # Note: (1 - Kresp_0) * gpp_mean = npp_mean, so this is equivalent to the original calculation
                npp_mean = steady_state_results['npp_mean']
                Cland_0 = npp_mean / args.Ksoil_0
                print(f"  Calculated Cland_0: {Cland_0:.4f} kg C m⁻²")
                print(f"  Verification: Ksoil_0 * Cland_0 = {args.Ksoil_0:.4f} * {Cland_0:.4f} = {args.Ksoil_0 * Cland_0:.4f}")
                print(f"  Target NPP: {npp_mean:.4f}")
                
                # Calculate Ktfp_0 from steady-state (using alpha=0.5 as initial guess)
                # At steady-state: GPP = Ktfp_0 * Cland_0 ** alpha
                # So: Ktfp_0 = GPP / (Cland_0 ** alpha)
                alpha_guess = 0.5
                Ktfp_0 = gpp_mean / (Cland_0 ** alpha_guess)
                print(f"  Calculated Ktfp_0: {Ktfp_0:.4f} (using alpha={alpha_guess})")
                print(f"  Verification: Ktfp_0 * Cland_0^{alpha_guess} = {Ktfp_0:.4f} * {Cland_0:.4f}^{alpha_guess} = {Ktfp_0 * (Cland_0 ** alpha_guess):.4f}")
                print(f"  Target GPP: {gpp_mean:.4f}")
                print(f"  Steady-state check: NPP = (1-Kresp_0) * GPP = {(1-Kresp_0):.4f} * {gpp_mean:.4f} = {(1-Kresp_0) * gpp_mean:.4f}")
                print(f"  Steady-state check: NPP = Ksoil_0 * Cland_0 = {args.Ksoil_0:.4f} * {Cland_0:.4f} = {args.Ksoil_0 * Cland_0:.4f}")
                
                # Build fixed parameters dictionary
                fixed_params = {
                    'Ksoil_0': args.Ksoil_0,
                    'Kresp_0': Kresp_0,
                    'Cland_init': Cland_0
                }
                
                # Check if alpha was provided on command line
                if args.alpha is not None:
                    # Use provided alpha, skip optimization
                    optimal_alpha = args.alpha
                    print(f"Using provided alpha: {optimal_alpha}")
                    print(f"Fixed parameters: Ksoil_0={args.Ksoil_0:.4f}, Kresp_0={Kresp_0:.4f}, Cland_0={Cland_0:.4f}")
                    print(f"User-specified alpha: {optimal_alpha}")
                else:
                    # Run analytical optimization
                    print(f"Fixed parameters: Ksoil_0={args.Ksoil_0:.4f}, Kresp_0={Kresp_0:.4f}, Ktfp_0={Ktfp_0:.4f}, Cland_0={Cland_0:.4f}")
                    print(f"Optimizing: alpha")
                    
                    # Calculate optimal alpha analytically for Step 1
                    print(f"\n=== Analytical Alpha Optimization for {region} / {model} ===")
                    optimal_alpha, mse_values = calculate_optimal_alpha_step1(
                        data_df, args.Ksoil_0, Kresp_0, Cland_0, alpha_bounds=(-1, 1)
                    )
                
                # Create parameter dictionary with optimal alpha
                param_dict = {
                    'Ksoil_0': args.Ksoil_0,
                    'Kresp_0': Kresp_0,
                    'Cland_init': Cland_0,
                    'alpha': optimal_alpha,
                    'Ktfp_0': gpp_mean / (Cland_0 ** optimal_alpha),  # Recalculate for optimal alpha
                    # Step 1 doesn't fit climate sensitivity parameters - set to 0 for inheritance
                    'Ktfp_tas0': 0.0,
                    'Ktfp_tas1': 0.0,
                    'Ktfp_pr0': 0.0,
                    'Ktfp_pr1': 0.0,
                    'region': region,
                    'model': model,
                    'step': 'step1'
                }
                
                print(f"DEBUG: Final parameters: {param_dict}")
                
                # Run final simulation to verify fit
                print(f"\n=== Running Final Simulation for Verification ===")
                print(f"DEBUG: Simulation parameters:")
                print(f"  Ksoil_0: {param_dict['Ksoil_0']:.6f}")
                print(f"  Kresp_0: {param_dict['Kresp_0']:.6f}")
                print(f"  Ktfp_0: {param_dict['Ktfp_0']:.6f}")
                print(f"  alpha: {param_dict['alpha']:.6f}")
                print(f"  Cland_init: {param_dict['Cland_init']:.6f}")
                print(f"  use_observed_npp_for_cland: True")
                results_df = run_bgc_simulation(data_df, param_dict, use_observed_npp_for_cland=True)
                
                success = True
                if args.alpha is not None:
                    # User-specified alpha
                    optimization_info = {
                        'success': True,
                        'method': 'user_specified_alpha',
                        'optimal_alpha': optimal_alpha,
                        'final_mse': 'N/A (user-specified)'
                    }
                else:
                    # Analytically optimized alpha
                    optimization_info = {
                        'success': True,
                        'method': 'analytical_alpha_optimization',
                        'optimal_alpha': optimal_alpha,
                        'final_mse': mse_values[optimal_alpha]
                    }
                
                if success:
                    successful_runs += 1
                    
                    # Add metadata to results (preserve optimized parameters)
                    param_dict['gpp_mean'] = gpp_mean
                    param_dict['npp_mean'] = steady_state_results['npp_mean']
                    
                    all_fitted_params.append(param_dict)
                    
                    # Save individual results
                    output_filename = get_output_filename("simulation_results", region, model, "step1")
                    output_filepath = os.path.join(get_run_output_directory(), output_filename)
                    results_df.to_csv(output_filepath, index=False)
                    print(f"Simulation results saved to {output_filepath}")
                    
                    if args.alpha is not None:
                        print(f"✅ Success: User-specified alpha used")
                        print(f"   Alpha: {param_dict.get('alpha', 'N/A')}")
                        print(f"   Method: {optimization_info.get('method', 'N/A')}")
                    else:
                        print(f"✅ Success: Analytical optimization completed")
                        print(f"   Optimized alpha: {param_dict.get('alpha', 'N/A')}")
                        print(f"   Final MSE: {optimization_info.get('final_mse', 'N/A')}")
                        print(f"   Method: {optimization_info.get('method', 'N/A')}")
                else:
                    failed_runs += 1
                    print(f"❌ Failed: {optimization_info.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed_runs += 1
                print(f"❌ Failed: {str(e)}")
    
    # Save all fitted parameters
    if all_fitted_params:
        save_fitted_parameters(all_fitted_params, "step1")
    
    # Print summary
    print(f"\n=== Step 1 Summary ===")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total combinations: {total_combinations}")
    
    return all_fitted_params, successful_runs, failed_runs
