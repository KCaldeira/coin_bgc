import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.optimize as optimize
import os
from datetime import datetime
import argparse

def setup_output_directory():
    """
    Create output directory if it doesn't exist and return the path.
    """
    output_dir = "data/output"
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory ready: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Error creating output directory: {e}")
        # Fallback to current directory if data/output fails
        fallback_dir = "output"
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"Using fallback output directory: {fallback_dir}")
        return fallback_dir

def get_output_filename(base_name, region, model, step="step1", extension=".csv"):
    """
    Generate a standardized output filename with timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{region}_{model}_{step}_{timestamp}{extension}"

def load_and_filter_data(filepath, region, model):
    # Columns to keep
    columns = ['year', 'tas', 'pr', 'gpp', 'npp', 'region', 'model']
    # Load CSV
    df = pd.read_csv(filepath, usecols=columns)
    # Filter for region and model
    filtered = df[(df['region'] == region) & (df['model'] == model)]
    return filtered


def run_bgc_simulation(filtered_df, params):
    # Sort by year
    filtered_df = filtered_df.sort_values('year').reset_index(drop=True)
    # print size of filtered_df along with name of region and model
    # print(f"Size of filtered_df: {filtered_df.shape}")
    # print(f"Region: {filtered_df['region'].iloc[0]}")
    # print(f"Model: {filtered_df['model'].iloc[0]}")
    years = filtered_df['year'].values
    alpha = params['alpha']  # dimensionless exponent for power law scaling of production with Cland
    Cland = params['Cland_init']
    # Store results
    results = []
    for i, row in filtered_df.iterrows():
        year = row['year']
        tas = row['tas']
        pr = row['pr']
        # Calculate Ksoil, Kresp, Ktfp as linear functions of tas and pr
        Ksoil = params['Ksoil_0'] + params['Ksoil_tas'] * tas + params['Ksoil_pr'] * pr
        Kresp = params['Kresp_0'] + params['Kresp_tas'] * tas + params['Kresp_pr'] * pr
        Ktfp  = params['Ktfp_0'] * (1 + params['Ktfp_tas']  * tas + params['Ktfp_pr']  * pr)
        GPP = Ktfp * (Cland ** alpha)
        Presp = Kresp * GPP # plant respiration
        NPP = GPP - Presp
        Sresp = Ksoil * Cland # soil respiration
        dCland_dt = NPP - Sresp
        results.append({
            'year': year,
            'Cland': Cland,
            'GPP': GPP,
            'NPP': NPP,
            'Presp': Presp,
            'Sresp': Sresp,
            'dCland_dt': dCland_dt,
            'tas_data': tas,
            'pr_data': pr,
            'gpp_data': row['gpp'],
            'npp_data': row['npp'],
            'region': row['region'],
            'model': row['model'],
            'Ksoil': Ksoil,
            'Kresp': Kresp,
            'Ktfp': Ktfp
        })
        # March forward
        Cland = Cland + dCland_dt  # dt = 1 year
    results_df = pd.DataFrame(results)
    print(".", end="", flush=True)
    return results_df

def first_guess_user_params(filtered_df, n_years, alpha, Ksoil):
    avg_start = filtered_df['year'].min()
    avg_end = avg_start + n_years - 1  # n_years=1 gives only the first year

    # compute Cland_init from NPP average and Ksoil
    avg_npp = filtered_df[(filtered_df['year'] >= avg_start) & (filtered_df['year'] <= avg_end)]['npp'].mean()
    Cland_init = avg_npp / Ksoil
    
    # compute Ktfp from Cland, alpha, and GPP
    avg_gpp = filtered_df[(filtered_df['year'] >= avg_start) & (filtered_df['year'] <= avg_end)]['gpp'].mean()
    Ktfp = avg_gpp / (Cland_init ** alpha)

    # --- Kresp regression with statsmodels for standard errors and p-values ---
    mask = (filtered_df['gpp'] > 0) & (filtered_df['npp'] > 0)
    df_reg = filtered_df[mask].copy()
    y = (df_reg['npp'] / df_reg['gpp']).values
    X = df_reg[['tas', 'pr']]
    X = sm.add_constant(X)  # Adds intercept
    model = sm.OLS(y, X).fit()
    Kresp_0 = model.params['const']
    Kresp_tas = model.params['tas']
    Kresp_pr = model.params['pr']
    print(f"Kresp_0: {Kresp_0}, Kresp_tas: {Kresp_tas}, Kresp_pr: {Kresp_pr}")
    print(model.summary())

    # now we will do similar with total factor productivity, under the assumption that Cland is constant
    mask = (filtered_df['gpp'] > 0) & (filtered_df['npp'] > 0)
    df_reg = filtered_df[mask].copy()
    y = (df_reg['gpp'] / Cland_init**alpha).values
    X = df_reg[['tas', 'pr']]
    X = sm.add_constant(X)  # Adds intercept
    model = sm.OLS(y, X).fit()
    Ktfp_0 = model.params['const']
    Ktfp_tas = model.params['tas']/Ktfp_0
    Ktfp_pr = model.params['pr']/Ktfp_0
    print(f"Ktfp_0: {Ktfp_0}, Ktfp_tas: {Ktfp_tas}, Ktfp_pr: {Ktfp_pr}")
    print(model.summary())

    return {
        'Ksoil_0': Ksoil,
        'Ksoil_tas': 0.0,
        'Ksoil_pr': 0.0,
        'Kresp_0': Kresp_0,
        'Kresp_tas': Kresp_tas,
        'Kresp_pr': Kresp_pr,
        'Ktfp_0': Ktfp_0,
        'Ktfp_tas': Ktfp_tas,
        'Ktfp_pr': Ktfp_pr,
        'alpha': alpha,
        'Cland_init': Cland_init
    }

def step1_picontrol_parameter_estimation(region, model, n_years=1, fixed_params=None):
    """
    Step 1: Estimate base parameters using piControl data.
    Assumes all _tas and _pr coefficients are zero.
    """
    # Load piControl data
    filepath = "data/input/Data_regression_piControl.csv"  # Use actual piControl file
    print(f"Loading data from: {filepath}")
    
    # First, let's see what's in the file
    try:
        full_df = pd.read_csv(filepath, usecols=['year', 'tas', 'pr', 'gpp', 'npp', 'region', 'model'])
        print(f"Full piControl data shape: {full_df.shape}")
        print(f"Available models: {full_df['model'].unique()}")
    except Exception as e:
        print(f"Error reading piControl file: {e}")
        return None, None
    
    filtered_df = load_and_filter_data(filepath, region, model)
    
    if filtered_df.empty:
        print(f"No data found for region={region}, model={model}")
        return None, None
    
    print(f"Step 1: Estimating base parameters for {region}, {model}")
    print(f"Data shape: {filtered_df.shape}")
    
    # Determine which parameters to optimize
    if fixed_params is None:
        fixed_params = {}
    
    # List of parameters that can be optimized
    optimizable_params = ['Ksoil_0', 'Kresp_0', 'Ktfp_0', 'alpha']
    params_to_optimize = [p for p in optimizable_params if p not in fixed_params]
    
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    print(f"Optimizing parameters: {params_to_optimize}")
    
    # Define objective function for optimization
    def objective_function(optimization_params):
        # Start with fixed parameters
        user_params = fixed_params.copy()
        
        # Add optimized parameters
        for i, param_name in enumerate(params_to_optimize):
            user_params[param_name] = optimization_params[i]
        
        # Set climate sensitivity parameters to zero (Step 1 assumption)
        user_params.update({
            'Ksoil_tas': 0.0,
            'Ksoil_pr': 0.0,
            'Kresp_tas': 0.0,
            'Kresp_pr': 0.0,
            'Ktfp_tas': 0.0,
            'Ktfp_pr': 0.0
        })
        
        # Build full params dictionary
        params_dict = build_params(filtered_df, user_params)
        
        # Run simulation
        results_df = run_bgc_simulation(filtered_df, params_dict)
        
        # Calculate error (difference between predicted and observed GPP/NPP)
        gpp_error = np.mean((results_df['GPP'] - results_df['gpp_data'])**2)
        npp_error = np.mean((results_df['NPP'] - results_df['npp_data'])**2)
        
        total_error = gpp_error + npp_error
        return total_error
    
    # Handle case where all parameters are fixed
    if not params_to_optimize:
        print("All parameters are fixed - no optimization needed")
        optimal_params = fixed_params.copy()
        optimal_params.update({
            'Ksoil_tas': 0.0, 'Ksoil_pr': 0.0,
            'Kresp_tas': 0.0, 'Kresp_pr': 0.0,
            'Ktfp_tas': 0.0, 'Ktfp_pr': 0.0
        })
        return optimal_params, filtered_df
    
    # Define parameter defaults and bounds
    param_defaults = {
        'Ksoil_0': 0.05,
        'Kresp_0': 0.4,
        'Ktfp_0': 1.0,
        'alpha': 0.3
    }
    
    param_bounds = {
        'Ksoil_0': (0.001, 1.0),
        'Kresp_0': (0.001, 0.99),
        'Ktfp_0': (0.001, 10.0),
        'alpha': (0.1, 0.8)
    }
    
    # Create initial guess and bounds for parameters to optimize
    initial_guess = [param_defaults[p] for p in params_to_optimize]
    bounds = [param_bounds[p] for p in params_to_optimize]
    
    # Optimize
    result = optimize.minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        # Extract optimized parameter values
        optimized_values = result.x
        
        # Estimate standard errors from Hessian inverse
        try:
            # For L-BFGS-B, hess_inv is a LinearOperator; convert to dense
            hess_inv = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv
            param_std = np.sqrt(np.diag(hess_inv))
            
            # Print standard errors for optimized parameters
            std_str = ", ".join([f"{params_to_optimize[i]}={param_std[i]:.4g}" for i in range(len(params_to_optimize))])
            print(f"Standard errors: {std_str}")
        except Exception as e:
            print(f"Could not compute standard errors: {e}")
            param_std = [None] * len(params_to_optimize)
        
        # Print optimal values for optimized parameters
        opt_str = ", ".join([f"{params_to_optimize[i]}={optimized_values[i]:.4f}" for i in range(len(params_to_optimize))])
        print(f"Optimization successful!")
        print(f"Optimal parameters: {opt_str}")
        print(f"Final error: {result.fun:.6f}")
        
        # Build complete parameter dictionary
        optimal_params = fixed_params.copy()
        
        # Add optimized parameters and their standard errors
        for i, param_name in enumerate(params_to_optimize):
            optimal_params[param_name] = optimized_values[i]
            optimal_params[f"{param_name}_stderr"] = param_std[i]
        
        # Add climate sensitivity parameters
        optimal_params.update({
            'Ksoil_tas': 0.0,
            'Ksoil_pr': 0.0,
            'Kresp_tas': 0.0,
            'Kresp_pr': 0.0,
            'Ktfp_tas': 0.0,
            'Ktfp_pr': 0.0
        })
        
        return optimal_params, filtered_df
    else:
        print(f"Optimization failed: {result.message}")
        return None, filtered_df

def build_params(filtered_df, user_params):
    """
    Build a complete parameter dictionary for the simulation.
    Computes Cland_init if not provided, using steady-state condition: Cland = NPP/Ksoil
    """
    params = user_params.copy()
    
    # Set default values for missing parameters
    defaults = {
        'Ksoil_0': 0.04,
        'Kresp_0': 0.5,
        'Ktfp_0': 1.0,
        'alpha': 0.3,
        'Ksoil_tas': 0.0,
        'Ksoil_pr': 0.0,
        'Kresp_tas': 0.0,
        'Kresp_pr': 0.0,
        'Ktfp_tas': 0.0,
        'Ktfp_pr': 0.0
    }
    
    # Fill in missing parameters with defaults
    for key, default_value in defaults.items():
        if key not in params or params[key] is None:
            params[key] = default_value
            print(f"Using default {key}: {default_value}")
    
    # If Cland_init is not provided, compute it from steady-state condition 
    if 'Cland_init' not in params or params['Cland_init'] is None:
        avg_npp = filtered_df['npp'].mean()
        Ksoil = params.get('Ksoil_0', 0.04)
        params['Cland_init'] = avg_npp / Ksoil if Ksoil != 0 else 1.0
    
    return params

def save_fitted_parameters(fitted_params_list, step="step1", single_file=True):
    """
    Save fitted parameters for all region/model combinations to a CSV file.
    Each row represents one region/model combination with all its parameters and standard errors.
    
    Args:
        fitted_params_list: List of parameter dictionaries
        step: Analysis step identifier
        single_file: If True, save all parameters to a single timestamped file
    """
    # Setup output directory
    output_dir = setup_output_directory()
    
    if not fitted_params_list:
        print("No parameters to save")
        return None
    
    # Convert list of parameter dictionaries to DataFrame
    params_df = pd.DataFrame(fitted_params_list)
    
    # Define all possible parameters
    all_params = [
        'region', 'model',
        'Ksoil_0', 'Ksoil_0_stderr',
        'Kresp_0', 'Kresp_0_stderr',
        'Ktfp_0', 'Ktfp_0_stderr',
        'alpha', 'alpha_stderr',
        'Ksoil_tas', 'Ksoil_pr',
        'Kresp_tas', 'Kresp_pr',
        'Ktfp_tas', 'Ktfp_pr'
    ]
    
    # Keep only the parameters that exist in the DataFrame
    available_params = [param for param in all_params if param in params_df.columns]
    params_df = params_df[available_params]
    
    if single_file:
        # Save all parameters to a single timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fitted_parameters_all_{step}_{timestamp}.csv"
        output_file = os.path.join(output_dir, filename)
        params_df.to_csv(output_file, index=False)
        print(f"All parameters saved to {output_file}")
        return params_df
    else:
        # Save individual files for each region/model combination
        for _, row in params_df.iterrows():
            region = row['region']
            model = row['model']
            filename = get_output_filename("fitted_parameters", region, model, step)
            output_file = os.path.join(output_dir, filename)
            row_df = pd.DataFrame([row])
            row_df.to_csv(output_file, index=False)
            print(f"Parameters for {region}/{model} saved to {output_file}")
        return params_df

def get_available_regions_and_models(filepath="data/input/Data_regression_piControl.csv"):
    """
    Get all available regions and models from the data file.
    """
    try:
        df = pd.read_csv(filepath, usecols=['region', 'model'])
        regions = sorted(df['region'].unique())
        models = sorted(df['model'].unique())
        return regions, models
    except Exception as e:
        print(f"Error reading data file: {e}")
        return [], []

def run_single_region_model(region, model, args, user_params=None):
    """
    Run simulation for a single region/model combination.
    
    Returns:
        tuple: (success, params_dict, results_df) where success is boolean
    """
    try:
        print(f"\n--- Processing {region} / {model} ---")
        
        # Determine which main parameters were provided by user
        main_params = ['Ksoil_0', 'Kresp_0', 'Ktfp_0', 'alpha']
        
        if user_params is None:
            user_params = {}
            # Add climate sensitivity parameters
            user_params.update({
                'Ksoil_tas': args.Ksoil_tas,
                'Ksoil_pr': args.Ksoil_pr,
                'Kresp_tas': args.Kresp_tas,
                'Kresp_pr': args.Kresp_pr,
                'Ktfp_tas': args.Ktfp_tas,
                'Ktfp_pr': args.Ktfp_pr
            })
        
        fixed_main_params = {key: value for key, value in user_params.items() if key in main_params and value is not None}
        
        if len(fixed_main_params) == len(main_params):
            print("All main parameters provided - running simulation directly...")
            # Load data
            filepath = "data/input/Data_regression_piControl.csv"
            filtered_df = load_and_filter_data(filepath, region, model)
            
            if filtered_df.empty:
                print(f"No data found for region={region}, model={model}")
                return False, None, None
            
            # Build complete parameter dictionary
            params_dict = build_params(filtered_df, user_params)
            
            # Run simulation with user parameters
            results_df = run_bgc_simulation(filtered_df, params_dict)
            
            # Add region and model info
            user_params['region'] = region
            user_params['model'] = model
            
            return True, user_params, results_df
            
        else:
            print("Running smart optimization...")
            if fixed_main_params:
                print(f"Fixed parameters: {list(fixed_main_params.keys())}")
                print(f"Will optimize: {[p for p in main_params if p not in fixed_main_params]}")
            else:
                print("Will optimize all main parameters")
            
            # Run optimization with fixed parameters
            optimal_params, filtered_df = step1_picontrol_parameter_estimation(
                region, model, args.n_years, fixed_params=fixed_main_params
            )
            
            if optimal_params is not None:
                # Add climate sensitivity parameters from user input
                optimal_params.update({
                    'Ksoil_tas': args.Ksoil_tas,
                    'Ksoil_pr': args.Ksoil_pr,
                    'Kresp_tas': args.Kresp_tas,
                    'Kresp_pr': args.Kresp_pr,
                    'Ktfp_tas': args.Ktfp_tas,
                    'Ktfp_pr': args.Ktfp_pr
                })
                
                # Add region and model info to parameters
                optimal_params['region'] = region
                optimal_params['model'] = model
                
                # Run simulation with optimal parameters
                params_dict = build_params(filtered_df, optimal_params)
                results_df = run_bgc_simulation(filtered_df, params_dict)
                
                return True, optimal_params, results_df
            else:
                print(f"Optimization failed for {region}/{model}")
                return False, None, None
                
    except Exception as e:
        print(f"Error processing {region}/{model}: {e}")
        return False, None, None

def run_complete_analysis(args, step="step1"):
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
        print("Step 2 not yet implemented")
        return
    elif step == "step3":
        print("Step 3 not yet implemented")
        return
    elif step == "all":
        print("Running all steps sequentially...")
        # Step 1
        all_fitted_params, successful_runs, failed_runs = run_step1_analysis(args, regions_to_run, models_to_run)
        # TODO: Add Steps 2 and 3 when implemented
    else:
        print(f"Unknown step: {step}")
        return
    
    # Print summary
    total_combinations = len(regions_to_run) * len(models_to_run)
    print(f"\n=== Summary ===")
    print(f"Total combinations processed: {total_combinations}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/total_combinations*100:.1f}%")
    print(f"Output directory: {output_dir}")
    
    if all_fitted_params:
        print(f"All fitted parameters saved to: fitted_parameters_all_{args.step}_*.csv")
    
    if failed_runs > 0:
        print(f"Warning: {failed_runs} runs failed. Check the output above for details.")

def setup_analysis_environment(args):
    """
    Setup output directories, validate data files, etc.
    """
    output_dir = setup_output_directory()
    return output_dir

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
    
    # Add climate sensitivity parameters (these have defaults, so always add them)
    user_params.update({
        'Ksoil_tas': args.Ksoil_tas,
        'Ksoil_pr': args.Ksoil_pr,
        'Kresp_tas': args.Kresp_tas,
        'Kresp_pr': args.Kresp_pr,
        'Ktfp_tas': args.Ktfp_tas,
        'Ktfp_pr': args.Ktfp_pr
    })
    
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
            
            # Run simulation for this region/model
            success, params_dict, results_df = run_single_region_model(region, model, args, user_params)
            
            if success:
                successful_runs += 1
                all_fitted_params.append(params_dict)
                
                # Save individual simulation results
                results_filename = get_output_filename("simulation_results", region, model, args.step)
                results_filepath = os.path.join(setup_output_directory(), results_filename)
                results_df.to_csv(results_filepath, index=False)
                print(f"Simulation results saved to {results_filepath}")
            else:
                failed_runs += 1
                print(f"Failed to process {region} / {model}")
    
    # Save all fitted parameters to a single file
    if all_fitted_params:
        save_fitted_parameters(all_fitted_params, args.step, single_file=True)
    
    return all_fitted_params, successful_runs, failed_runs

def collect_and_save_results(all_fitted_params, step, args):
    """
    Collect results from all region/model combinations and save.
    """
    if all_fitted_params:
        save_fitted_parameters(all_fitted_params, step, single_file=True)
    return len(all_fitted_params)

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
    parser.add_argument('--n_years', type=int, default=1,
                       help='Number of years for parameter estimation')
    
    # Model parameters (optional - will use optimization if not provided)
    parser.add_argument('--Ksoil_0', type=float, default=None,
                       help='Base soil respiration rate (if None, will be optimized)')
    parser.add_argument('--Kresp_0', type=float, default=None,
                       help='Base plant respiration fraction (if None, will be optimized)')
    parser.add_argument('--Ktfp_0', type=float, default=None,
                       help='Base total factor productivity (if None, will be optimized)')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Power law exponent for production scaling (if None, will be optimized)')
    
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
    parser.add_argument('--step', type=str, default='step1',
                       help='Analysis step to run: step1, step2, step3, or all')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_command_line_args()
    
    # Run the complete analysis
    run_complete_analysis(args, args.step)