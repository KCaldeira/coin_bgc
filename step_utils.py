import pandas as pd
import numpy as np
import os
from datetime import datetime
import scipy.optimize as optimize
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Global variable to store the current run's timestamp
_current_run_timestamp = None

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

def get_run_output_directory():
    """
    Get the timestamped output directory for the current run.
    Creates a new timestamped subdirectory if one doesn't exist.
    """
    global _current_run_timestamp
    
    if _current_run_timestamp is None:
        _current_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_output_dir = setup_output_directory()
    run_output_dir = os.path.join(base_output_dir, f"run_{_current_run_timestamp}")
    
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        return run_output_dir
    except Exception as e:
        print(f"Error creating run output directory: {e}")
        return base_output_dir

def reset_run_timestamp():
    """
    Reset the run timestamp to create a new run directory.
    """
    global _current_run_timestamp
    _current_run_timestamp = None

def get_most_recent_output_directory():
    """
    Find the most recent timestamped output directory.
    Returns the path to the most recent run directory, or None if none exist.
    """
    base_output_dir = setup_output_directory()
    
    # Find all run directories
    run_dirs = []
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path) and item.startswith("run_"):
            run_dirs.append(item_path)
    
    if not run_dirs:
        return None
    
    # Sort by creation time (most recent first)
    run_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return run_dirs[0]

def get_output_filename(base_name, region, model, step="step1", extension=".csv"):
    """
    Generate a standardized output filename without timestamp.
    """
    return f"{base_name}_{region}_{model}_{step}{extension}"

def load_and_filter_data(filepath, region, model):
    """
    Load and filter data for a specific region and model.
    """
    # Columns to keep
    columns = ['year', 'tas', 'pr', 'gpp', 'npp', 'region', 'model']
    # Load CSV
    df = pd.read_csv(filepath, usecols=columns)
    # Filter for region and model
    filtered = df[(df['region'] == region) & (df['model'] == model)]
    
    # Remove rows where year is NaN (invalid data)
    original_count = len(filtered)
    filtered = filtered.dropna(subset=['year'])
    if len(filtered) < original_count:
        print(f"Warning: Removed {original_count - len(filtered)} rows with NaN year values from {filepath}")
    
    # Check for data quality issues
    tas_none_mask = filtered['tas'].isna() | (filtered['tas'] == None)
    pr_none_mask = filtered['pr'].isna() | (filtered['pr'] == None)
    
    if tas_none_mask.any() or pr_none_mask.any():
        # Find the specific rows with missing data
        tas_missing = filtered[tas_none_mask]
        pr_missing = filtered[pr_none_mask]
        
        print(f"ERROR: Missing temperature (tas) data in {filepath}")
        if not tas_missing.empty:
            print(f"  Missing tas values for {len(tas_missing)} rows:")
            for _, row in tas_missing.iterrows():
                print(f"    Year: {row['year']}, Region: {row['region']}, Model: {row['model']}, tas value: {row['tas']}")
        
        print(f"ERROR: Missing precipitation (pr) data in {filepath}")
        if not pr_missing.empty:
            print(f"  Missing pr values for {len(pr_missing)} rows:")
            for _, row in pr_missing.iterrows():
                print(f"    Year: {row['year']}, Region: {row['region']}, Model: {row['model']}, pr value: {row['pr']}")
        
        raise ValueError(f"Data quality issue: Missing tas/pr values in {filepath} for {region}/{model}")
    
    return filtered

def load_and_filter_data_step2(region, model):
    """
    Load and concatenate historical and SSP585-bgc data for Step 2.
    """
    try:
        # Columns to keep
        columns = ['year', 'tas', 'pr', 'gpp', 'npp', 'region', 'model']
        
        # Load historical biogeochemical data
        hist_file = "data/input/Data_regression_hist-bgc.csv"
        hist_df = pd.read_csv(hist_file, usecols=columns)
        hist_filtered = hist_df[(hist_df['region'] == region) & (hist_df['model'] == model)].copy()
        
        # Load SSP585-bgc data
        ssp585_file = "data/input/Data_regression_ssp585-bgc.csv"
        ssp585_df = pd.read_csv(ssp585_file, usecols=columns)
        ssp585_filtered = ssp585_df[(ssp585_df['region'] == region) & (ssp585_df['model'] == model)].copy()
        
        # Concatenate the two datasets
        combined_df = pd.concat([hist_filtered, ssp585_filtered], ignore_index=True)
        
        # Remove rows where year is NaN (invalid data)
        original_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['year'])
        if len(combined_df) < original_count:
            print(f"Warning: Removed {original_count - len(combined_df)} rows with NaN year values from Step 2 concatenated data")
        
        # Sort by year to ensure chronological order
        combined_df = combined_df.sort_values('year').reset_index(drop=True)
        
        # Check for data quality issues
        tas_none_mask = combined_df['tas'].isna() | (combined_df['tas'] == None)
        pr_none_mask = combined_df['pr'].isna() | (combined_df['pr'] == None)
        
        if tas_none_mask.any() or pr_none_mask.any():
            # Find the specific rows with missing data
            tas_missing = combined_df[tas_none_mask]
            pr_missing = combined_df[pr_none_mask]
            
            print(f"ERROR: Missing temperature (tas) data in Step 2 concatenated data")
            if not tas_missing.empty:
                print(f"  Missing tas values for {len(tas_missing)} rows:")
                for _, row in tas_missing.iterrows():
                    print(f"    Year: {row['year']}, Region: {row['region']}, Model: {row['model']}, tas value: {row['tas']}")
            
            print(f"ERROR: Missing precipitation (pr) data in Step 2 concatenated data")
            if not pr_missing.empty:
                print(f"  Missing pr values for {len(pr_missing)} rows:")
                for _, row in pr_missing.iterrows():
                    print(f"    Year: {row['year']}, Region: {row['region']}, Model: {row['model']}, pr value: {row['pr']}")
            
            raise ValueError(f"Data quality issue: Missing tas/pr values in Step 2 data for {region}/{model}")
        
        print(f"Loaded {len(hist_filtered)} historical + {len(ssp585_filtered)} SSP585-bgc records for {region}/{model}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error loading Step 2 data for {region}/{model}: {e}")
        return pd.DataFrame()

def load_and_filter_data_step3(region, model):
    """
    Load and concatenate historical and SSP585 data for Step 3.
    """
    try:
        # Columns to keep
        columns = ['year', 'tas', 'pr', 'gpp', 'npp', 'region', 'model']
        
        # Load historical data
        hist_file = "data/input/Data_regression_historical.csv"
        hist_df = pd.read_csv(hist_file, usecols=columns)
        hist_filtered = hist_df[(hist_df['region'] == region) & (hist_df['model'] == model)].copy()
        
        # Load SSP585 data
        ssp585_file = "data/input/Data_regression_ssp585.csv"
        ssp585_df = pd.read_csv(ssp585_file, usecols=columns)
        ssp585_filtered = ssp585_df[(ssp585_df['region'] == region) & (ssp585_df['model'] == model)].copy()
        
        # Concatenate the two datasets
        combined_df = pd.concat([hist_filtered, ssp585_filtered], ignore_index=True)
        
        # Remove rows where year is NaN (invalid data)
        original_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['year'])
        if len(combined_df) < original_count:
            print(f"Warning: Removed {original_count - len(combined_df)} rows with NaN year values from Step 3 concatenated data")
        
        # Sort by year to ensure chronological order
        combined_df = combined_df.sort_values('year').reset_index(drop=True)
        
        # Check for data quality issues
        tas_none_mask = combined_df['tas'].isna() | (combined_df['tas'] == None)
        pr_none_mask = combined_df['pr'].isna() | (combined_df['pr'] == None)
        
        if tas_none_mask.any() or pr_none_mask.any():
            # Find the specific rows with missing data
            tas_missing = combined_df[tas_none_mask]
            pr_missing = combined_df[pr_none_mask]
            
            print(f"ERROR: Missing temperature (tas) data in Step 3 concatenated data")
            if not tas_missing.empty:
                print(f"  Missing tas values for {len(tas_missing)} rows:")
                for _, row in tas_missing.iterrows():
                    print(f"    Year: {row['year']}, Region: {row['region']}, Model: {row['model']}, tas value: {row['tas']}")
            
            print(f"ERROR: Missing precipitation (pr) data in Step 3 concatenated data")
            if not pr_missing.empty:
                print(f"  Missing pr values for {len(pr_missing)} rows:")
                for _, row in pr_missing.iterrows():
                    print(f"    Year: {row['year']}, Region: {row['region']}, Model: {row['model']}, pr value: {row['pr']}")
            
            raise ValueError(f"Data quality issue: Missing tas/pr values in Step 3 data for {region}/{model}")
        
        print(f"Loaded {len(hist_filtered)} historical + {len(ssp585_filtered)} SSP585 records for {region}/{model}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error loading Step 3 data for {region}/{model}: {e}")
        return pd.DataFrame()

def load_co2_data(filepath="data/input/historical-ssp585_co2.csv"):
    """
    Load CO2 concentration data from the historical-SSP585 file.
    """
    try:
        co2_df = pd.read_csv(filepath)
        # Rename column for consistency
        co2_df = co2_df.rename(columns={'pco2 (ppm)': 'co2'})
        return co2_df
    except Exception as e:
        print(f"Error loading CO2 data: {e}")
        return None

def get_co2_for_year(co2_df, year):
    """
    Get CO2 concentration for a specific year.
    """
    if co2_df is None:
        return 284.318604  # Default pre-industrial value
    year_data = co2_df[co2_df['year'] == year]
    if len(year_data) > 0:
        return year_data['co2'].iloc[0]
    else:
        # Interpolate if year not found
        co2_df_sorted = co2_df.sort_values('year')
        return np.interp(year, co2_df_sorted['year'], co2_df_sorted['co2'])

def run_bgc_simulation(filtered_df, params, co2_df=None, co2_0=284.318604):
    """
    Run BGC simulation with optional CO2 dependence.
    
    Args:
        filtered_df: Climate and vegetation data
        params: Model parameters
        co2_df: CO2 concentration data (optional)
        co2_0: Reference CO2 concentration (default: pre-industrial)
    """
    # Sort by year
    filtered_df = filtered_df.sort_values('year').reset_index(drop=True)
    
    # Check for None/NaN values in critical columns
    tas_none_mask = filtered_df['tas'].isna() | (filtered_df['tas'] == None)
    pr_none_mask = filtered_df['pr'].isna() | (filtered_df['pr'] == None)
    
    if tas_none_mask.any() or pr_none_mask.any():
        # Find the specific rows with missing data
        tas_missing = filtered_df[tas_none_mask]
        pr_missing = filtered_df[pr_none_mask]
        
        print(f"ERROR: Missing temperature (tas) data in simulation data")
        if not tas_missing.empty:
            print(f"  Missing tas values for {len(tas_missing)} rows:")
            for _, row in tas_missing.iterrows():
                print(f"    Year: {row['year']}, Region: {row['region']}, Model: {row['model']}, tas value: {row['tas']}")
        
        print(f"ERROR: Missing precipitation (pr) data in simulation data")
        if not pr_missing.empty:
            print(f"  Missing pr values for {len(pr_missing)} rows:")
            for _, row in pr_missing.iterrows():
                print(f"    Year: {row['year']}, Region: {row['region']}, Model: {row['model']}, pr value: {row['pr']}")
        
        raise ValueError("Data quality issue: Missing tas/pr values in simulation data")
    
    years = filtered_df['year'].values
    alpha = params.get('alpha', 0.5)  # dimensionless exponent for power law scaling of production with Cland
    Cland = params.get('Cland_init', 1.0)
    
    # Store results
    results = []
    for i, row in filtered_df.iterrows():
        year = row['year']
        tas = row['tas']
        pr = row['pr']
        
        # Verify tas and pr are not None/NaN (should have been caught earlier)
        if tas is None or pd.isna(tas):
            raise ValueError(f"Temperature (tas) is None/NaN for year {year}, region {row['region']}, model {row['model']}, tas value: {tas}")
        if pr is None or pd.isna(pr):
            raise ValueError(f"Precipitation (pr) is None/NaN for year {year}, region {row['region']}, model {row['model']}, pr value: {pr}")
        
        # Additional check: ensure tas and pr are numeric
        try:
            tas = float(tas)
            pr = float(pr)
        except (ValueError, TypeError):
            raise ValueError(f"Non-numeric tas or pr value for year {year}, region {row['region']}, model {row['model']}, tas: {tas}, pr: {pr}")
        
        # Get CO2 concentration for this year
        co2 = get_co2_for_year(co2_df, year) if co2_df is not None else co2_0
        
        # Calculate Ksoil, Kresp (no climate sensitivity - removed those parameters)
        Ksoil = params.get('Ksoil_0', 0.1)
        Kresp = params.get('Kresp_0', 0.5)
        
        # Calculate Ktfp with optional CO2 dependence
        Ktfp_0 = params.get('Ktfp_0', 1.0)
        Ktfp_tas0 = params.get('Ktfp_tas0', 20.57)  # mean temperature (should be initialized from piControl)
        Ktfp_tas1 = params.get('Ktfp_tas1', 0.0) # linear temperature sensitivity
        Ktfp_pr0 = params.get('Ktfp_pr0', 3.26)  # mean precipitation (should be initialized from piControl)
        Ktfp_pr1 = params.get('Ktfp_pr1', 0.0) # linear precipitation sensitivity
    
        tas_factor = 1 + Ktfp_tas1 * (tas - Ktfp_tas0) 
        pr_factor = 1 + Ktfp_pr1 * (pr - Ktfp_pr0)

        if 'Ktfp_co2' in params and co2_df is not None:
            # CO2-dependent Ktfp calculation
            co2_factor = (1 + params['Ktfp_co2']) * ((co2/co2_0) / (params['Ktfp_co2'] + co2/co2_0))
        else:
            co2_factor = 1.0

        Ktfp = Ktfp_0 * tas_factor * pr_factor * co2_factor
        
        GPP = Ktfp * (Cland ** alpha)
        Presp = Kresp * GPP  # plant respiration
        NPP = GPP - Presp
        Sresp = Ksoil * Cland  # soil respiration
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
            'co2': co2,
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

def first_guess_user_params(filtered_df, alpha, Ksoil):
    """
    Generate first guess parameters for optimization.
    """
    # compute Cland_init from NPP average and Ksoil (using all available years)
    avg_npp = filtered_df['npp'].mean()
    Cland_init = avg_npp / Ksoil
    
    return Cland_init

def objective_function(params, filtered_df, param_names, user_params):
    """
    Objective function for parameter optimization.
    """
    # Create parameter dictionary
    param_dict = dict(zip(param_names, params))
    
    # Add user-provided parameters
    param_dict.update(user_params)
    
    # Get alpha and Ksoil from param_dict (with defaults)
    alpha = param_dict.get('alpha', 0.5)
    Ksoil = param_dict.get('Ksoil_0', 0.1)
    
    # Calculate Cland_init
    param_dict['Cland_init'] = first_guess_user_params(filtered_df, alpha, Ksoil)
    
    # Run simulation
    # Note: co2_df is not available in objective_function, so we need to pass it through user_params
    co2_df = user_params.get('_co2_df', None)
    
    # Quick check to ensure we have valid data
    if filtered_df.empty:
        print("ERROR: filtered_df is empty in objective_function")
        return 1e6  # Return high error
    
    # Check for any None values in tas or pr
    if filtered_df['tas'].isna().any() or filtered_df['pr'].isna().any():
        print("ERROR: Found NaN values in tas or pr in objective_function")
        return 1e6  # Return high error
    
    results_df = run_bgc_simulation(filtered_df, param_dict, co2_df)
    
    # Calculate error (MSE between simulated and observed NPP)
    mse = np.mean((results_df['NPP'] - results_df['npp_data'])**2)
    
    return mse

def run_single_region_model(region, model, args, user_params, co2_df=None):
    """
    Run simulation for a single region/model combination.
    """
    try:
        # Load data and determine actual step
        actual_step = args.step
        if args.step == "step1":
            data_file = "data/input/Data_regression_piControl.csv"
            filtered_df = load_and_filter_data(data_file, region, model)
        elif args.step == "step2":
            # For step2, concatenate historical and SSP585-bgc data
            filtered_df = load_and_filter_data_step2(region, model)
        elif args.step == "step3":
            # For step3, concatenate historical and SSP585 data
            filtered_df = load_and_filter_data_step3(region, model)
        elif args.step == "all":
            # For "all", determine step based on data file or context
            # This will be determined by the calling function
            data_file = "data/input/Data_regression_piControl.csv"  # default
            filtered_df = load_and_filter_data(data_file, region, model)
        else:
            data_file = "data/input/Data_regression_piControl.csv"  # default
            filtered_df = load_and_filter_data(data_file, region, model)
        
        if filtered_df.empty:
            print(f"No data found for {region} / {model}")
            return False, {}, pd.DataFrame()
        
        # Check if user provided all main parameters AND we're not in step2 or step3 (where we want to optimize additional parameters)
        main_params = ['Ksoil_0', 'Kresp_0', 'Ktfp_0', 'alpha']
        provided_main_params = [p for p in main_params if p in user_params and user_params[p] is not None]
        

        
        # For step2, we want to optimize Ktfp_co2 even if all main parameters are provided
        # For step3, we want to optimize climate sensitivity parameters even if all main parameters are provided
        if len(provided_main_params) == len(main_params) and args.step == "step1":
            # Use provided parameters (for step1 only)
            param_dict = user_params.copy()
            param_dict['Cland_init'] = first_guess_user_params(filtered_df, 
                                                             param_dict['alpha'], param_dict['Ksoil_0'])
            
            # Run simulation with provided parameters
            results_df = run_bgc_simulation(filtered_df, param_dict, co2_df)
            
            # Add metadata
            param_dict.update({
                'region': region,
                'model': model,
                'step': args.step
            })
            
            return True, param_dict, results_df
        
        # Optimize parameters
        # Define parameter bounds and initial guesses
        param_bounds = []
        param_names = []
        initial_guess = []
        
        # Add parameters to optimize (only those not provided by user)
        if 'Ksoil_0' not in user_params:
            param_names.append('Ksoil_0')
            param_bounds.append((0.001, 2.0))  # Expanded bounds
            initial_guess.append(0.1)
        
        if 'Kresp_0' not in user_params:
            param_names.append('Kresp_0')
            param_bounds.append((0.01, 0.99))
            initial_guess.append(0.5)
        
        if 'Ktfp_0' not in user_params:
            param_names.append('Ktfp_0')
            param_bounds.append((0.1, 50.0))  # Expanded bounds
            initial_guess.append(1.0)
        
        if 'alpha' not in user_params:
            param_names.append('alpha')
            param_bounds.append((0.05, 2.0))  # Expanded bounds
            initial_guess.append(0.5)
        
        # Add CO2 parameter for step2
        if args.step == "step2" and 'Ktfp_co2' not in user_params:
            param_names.append('Ktfp_co2')
            param_bounds.append((0.0, 2000.0))
            initial_guess.append(0.1)
        
        # Add climate sensitivity parameters for step3
        if args.step == "step3":
            climate_params = ['Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_pr0', 'Ktfp_pr1']
            for param in climate_params:
                # Only optimize climate sensitivity parameters that are not in user_params (i.e., not set to a specific value)
                if param not in user_params:
                    param_names.append(param)
                    if 'tas0' in param or 'pr0' in param:
                        # Reference values (temperature/precipitation means)
                        if 'tas0' in param:
                            param_bounds.append((10.0, 30.0))  # Temperature reference bounds
                            initial_guess.append(20.57)
                        else:  # pr0
                            param_bounds.append((1.0, 10.0))   # Precipitation reference bounds
                            initial_guess.append(3.26)
                    else:
                        # Sensitivity coefficients
                        param_bounds.append((-0.99, 0.99))  # Sensitivity bounds
                        if 'tas1' in param:
                            initial_guess.append(0.1)   # Temperature sensitivity
                        else:  # pr1
                            initial_guess.append(-0.05) # Precipitation sensitivity
        
        if not param_names:
            print("No parameters to optimize")
            return False, {}, pd.DataFrame()
        

        
        # Add CO2 data to user_params for objective function
        if co2_df is not None:
            user_params['_co2_df'] = co2_df
        else:
            # Ensure _co2_df is always present in user_params
            user_params['_co2_df'] = None
        
        # Run optimization
        print(f"DEBUG: Starting optimization with {len(param_names)} parameters")
        print(f"DEBUG: Initial guess: {initial_guess}")
        print(f"DEBUG: Initial objective value: {objective_function(initial_guess, filtered_df, param_names, user_params)}")
        
        result = optimize.minimize(
            objective_function,
            initial_guess,
            args=(filtered_df, param_names, user_params),
            bounds=param_bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-8}  # Increase iterations and tolerance
        )
        
        print(f"DEBUG: Optimization finished. Success: {result.success}")
        print(f"DEBUG: Final objective value: {result.fun}")
        print(f"DEBUG: Number of iterations: {result.nit}")
        print(f"DEBUG: Number of function evaluations: {result.nfev}")
        print(f"DEBUG: Final parameter values: {result.x}")
        
        if result.success:
            # Create parameter dictionary
            param_dict = dict(zip(param_names, result.x))
            param_dict.update(user_params)  # Add user-provided parameters
            
            # Add derived parameters
            param_dict['Cland_init'] = first_guess_user_params(filtered_df, 
                                                             param_dict['alpha'], param_dict['Ksoil_0'])
            
            # Run final simulation
            results_df = run_bgc_simulation(filtered_df, param_dict, co2_df)
            
            # Add metadata
            param_dict.update({
                'region': region,
                'model': model,
                'step': args.step,
                'optimization_success': True,
                'final_mse': result.fun
            })
            
            return True, param_dict, results_df
        else:
            print(f"Optimization failed for {region} / {model}: {result.message}")
            return False, {}, pd.DataFrame()
            
    except Exception as e:
        print(f"Error processing {region} / {model}: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, pd.DataFrame()

def optimize_parameters(fixed_params, params_to_optimize, data_df, co2_df=None):
    """
    Optimize parameters for BGC simulation using a clean, explicit approach.
    
    Args:
        fixed_params (dict): Dictionary of parameter names and their fixed values
        params_to_optimize (list): List of parameter names to optimize
        data_df (pd.DataFrame): Data for fitting (must contain year, tas, pr, npp columns)
        co2_df (pd.DataFrame, optional): CO2 concentration data
    
    Returns:
        tuple: (success, optimized_params, results_df, optimization_info)
            - success (bool): Whether optimization was successful
            - optimized_params (dict): Complete parameter dictionary with optimized values
            - results_df (pd.DataFrame): Simulation results
            - optimization_info (dict): Optimization details (iterations, final MSE, etc.)
    """
    print(f"DEBUG: Fixed parameters dictionary keys: {list(fixed_params.keys())}")
    print(f"DEBUG: Fixed parameters values: {fixed_params}")
    print(f"DEBUG: Parameters to optimize: {params_to_optimize}")
    print(f"DEBUG: Total parameters being handled: {len(fixed_params) + len(params_to_optimize)}")
    try:
        if data_df.empty:
            print("ERROR: data_df is empty")
            return False, {}, pd.DataFrame(), {}
        
        # Check for required columns
        required_columns = ['year', 'tas', 'pr', 'npp']
        missing_columns = [col for col in required_columns if col not in data_df.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns in data_df: {missing_columns}")
            return False, {}, pd.DataFrame(), {}
        
        # Define parameter bounds and initial guesses for optimization
        param_bounds = []
        initial_guess = []
        
        # Parameter definitions with bounds and initial guesses
        param_definitions = {
            'Ksoil_0': {'bounds': (0.001, 2.0), 'initial': 0.1},      # Expanded bounds for diverse ecosystems
            'Kresp_0': {'bounds': (0.01, 0.99), 'initial': 0.5},
            'Ktfp_0': {'bounds': (0.1, 50.0), 'initial': 1.0},        # Expanded bounds for diverse productivity
            'alpha': {'bounds': (0.05, 2.0), 'initial': 0.5},          # Expanded bounds for different production functions
            'Ktfp_co2': {'bounds': (0.0, 2000.0), 'initial': 0.1},
            'Ktfp_tas0': {'bounds': (10.0, 30.0), 'initial': 20.57},  # Reference temperature (°C)
            'Ktfp_tas1': {'bounds': (-0.99, 0.99), 'initial': 0.1},   # Temperature sensitivity
            'Ktfp_pr0': {'bounds': (1.0, 10.0), 'initial': 3.26},     # Reference precipitation (mm/day)
            'Ktfp_pr1': {'bounds': (-0.99, 0.99), 'initial': -0.05}   # Precipitation sensitivity
        }
        
        # Build optimization lists
        for param in params_to_optimize:
            if param in param_definitions:
                param_bounds.append(param_definitions[param]['bounds'])
                initial_guess.append(param_definitions[param]['initial'])
            else:
                print(f"WARNING: Unknown parameter '{param}' in params_to_optimize")
                # Use default bounds and initial guess for unknown parameters
                param_bounds.append((-1.0, 1.0))
                initial_guess.append(0.0)
        
        if not params_to_optimize:
            print("No parameters to optimize - running simulation with fixed parameters only")
            
            # Create complete parameter dictionary with fixed values
            complete_params = fixed_params.copy()
            
            # Add derived parameters
            alpha = complete_params.get('alpha', 0.5)
            Ksoil = complete_params.get('Ksoil_0', 0.1)
            complete_params['Cland_init'] = first_guess_user_params(data_df, alpha, Ksoil)
            
            # Run simulation
            results_df = run_bgc_simulation(data_df, complete_params, co2_df)
            
            # Create optimization info for no-optimization case
            optimization_info = {
                'success': True,
                'iterations': 0,
                'function_evaluations': 0,
                'initial_mse': 0.0,
                'final_mse': 0.0,
                'optimized_parameters': [],
                'fixed_parameters': list(fixed_params.keys())
            }
            
            return True, complete_params, results_df, optimization_info
        
        # Create complete parameter dictionary (fixed + optimized)
        complete_params = fixed_params.copy()
        
        # Add CO2 data to complete_params for objective function
        if co2_df is not None:
            complete_params['_co2_df'] = co2_df
        else:
            complete_params['_co2_df'] = None
        
        # Run optimization
        print(f"DEBUG: Starting optimization with {len(params_to_optimize)} parameters")
        print(f"DEBUG: Parameters to optimize: {params_to_optimize}")
        print(f"DEBUG: Fixed parameters: {list(fixed_params.keys())}")
        print(f"DEBUG: Initial guess: {initial_guess}")
        
        initial_mse = objective_function(initial_guess, data_df, params_to_optimize, complete_params)
        print(f"DEBUG: Initial objective value: {initial_mse}")
        
        result = optimize.minimize(
            objective_function,
            initial_guess,
            args=(data_df, params_to_optimize, complete_params),
            bounds=param_bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-8}
        )
        
        print(f"DEBUG: Optimization finished. Success: {result.success}")
        print(f"DEBUG: Final objective value: {result.fun}")
        print(f"DEBUG: Number of iterations: {result.nit}")
        print(f"DEBUG: Number of function evaluations: {result.nfev}")
        print(f"DEBUG: Final parameter values: {result.x}")
        
        if result.success:
            # Create complete parameter dictionary with optimized values
            optimized_params = fixed_params.copy()
            optimized_params.update(dict(zip(params_to_optimize, result.x)))
            
            # Add derived parameters
            alpha = optimized_params.get('alpha', 0.5)
            Ksoil = optimized_params.get('Ksoil_0', 0.1)
            optimized_params['Cland_init'] = first_guess_user_params(data_df, alpha, Ksoil)
            
            # Run final simulation
            results_df = run_bgc_simulation(data_df, optimized_params, co2_df)
            
            # Create optimization info
            optimization_info = {
                'success': True,
                'iterations': result.nit,
                'function_evaluations': result.nfev,
                'initial_mse': initial_mse,
                'final_mse': result.fun,
                'optimized_parameters': params_to_optimize,
                'fixed_parameters': list(fixed_params.keys())
            }
            
            return True, optimized_params, results_df, optimization_info
        else:
            print(f"Optimization failed: {result.message}")
            optimization_info = {
                'success': False,
                'error_message': result.message,
                'iterations': result.nit,
                'function_evaluations': result.nfev,
                'initial_mse': initial_mse,
                'final_mse': result.fun
            }
            return False, {}, pd.DataFrame(), optimization_info
            
    except Exception as e:
        print(f"Error in optimize_parameters: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, pd.DataFrame(), {'success': False, 'error': str(e)}

def save_fitted_parameters(all_fitted_params, step, single_file=True):
    """
    Save fitted parameters to CSV file(s) with standardized column order.
    """
    if not all_fitted_params:
        return
    
    output_dir = get_run_output_directory()
    # Ensure the run timestamp is set
    if _current_run_timestamp is None:
        get_run_output_directory()
    
    # Use the run timestamp instead of generating a new one
    timestamp = _current_run_timestamp
    
    # Define standardized column order
    # Alphanumeric columns first, then final_mse, then all parameters
    standard_columns = [
        'step', 'model', 'region', 'optimization_success',  # Alphanumeric columns first
        'final_mse',  # Then final_mse
        # All parameters in consistent order (Cland_init between alpha and Ktfp_co2)
        'Ksoil_0', 'Kresp_0', 'Ktfp_0', 'alpha', 'Cland_init', 'Ktfp_co2',
        'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_pr0', 'Ktfp_pr1'
    ]
    
    if single_file:
        # Save all parameters to one file
        all_params_df = pd.DataFrame(all_fitted_params)
        
        # Ensure all standard columns exist (set to 0 if missing)
        for col in standard_columns:
            if col not in all_params_df.columns:
                all_params_df[col] = 0.0
        
        # Reorder columns to match standard order
        existing_columns = [col for col in standard_columns if col in all_params_df.columns]
        all_params_df = all_params_df[existing_columns]
        
        filename = f"fitted_parameters_all_{step}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        all_params_df.to_csv(filepath, index=False)
        print(f"All fitted parameters saved to: {filepath}")
    else:
        # Save individual files (legacy option)
        for i, params in enumerate(all_fitted_params):
            region = params.get('region', f'region_{i}')
            model = params.get('model', f'model_{i}')
            
            # Ensure all standard columns exist in params
            for col in standard_columns:
                if col not in params:
                    params[col] = 0.0
            
            # Create DataFrame with standard column order
            params_df = pd.DataFrame([params])
            existing_columns = [col for col in standard_columns if col in params_df.columns]
            params_df = params_df[existing_columns]
            
            filename = f"fitted_parameters_{region}_{model}_{step}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            params_df.to_csv(filepath, index=False)

def get_available_regions_and_models():
    """
    Get all available regions and models from the data files.
    """
    try:
        # Try to load from piControl data first
        data_file = "data/input/Data_regression_piControl.csv"
        df = pd.read_csv(data_file, usecols=['region', 'model'])
        all_regions = sorted(df['region'].unique())
        all_models = sorted(df['model'].unique())
        return all_regions, all_models
    except Exception as e:
        print(f"Error reading data file: {e}")
        return [], []

def load_step_parameters_from_file(filepath):
    """
    Load parameters from a step output file.
    
    Args:
        filepath: Path to step fitted parameters file
    
    Returns:
        dict: Dictionary with (region, model) as keys and parameter dict as values
    """
    try:
        if not os.path.exists(filepath):
            print(f"Step output file not found: {filepath}")
            return {}
        
        step_df = pd.read_csv(filepath)
        step_params = {}
        
        for _, row in step_df.iterrows():
            region = row['region']
            model = row['model']
            step_params[(region, model)] = row.to_dict()
        
        print(f"Loaded step parameters for {len(step_params)} region/model combinations from {filepath}")
        return step_params
        
    except Exception as e:
        print(f"Error loading step parameters: {e}")
        return {}

def run_single_region_model_clean(region, model, step, fixed_params, params_to_optimize, co2_df=None):
    """
    Run simulation for a single region/model combination using the clean parameter approach.
    
    Args:
        region (str): Geographic region
        model (str): Climate model
        step (str): Analysis step ('step1', 'step2', 'step3')
        fixed_params (dict): Dictionary of fixed parameter values
        params_to_optimize (list): List of parameter names to optimize
        co2_df (pd.DataFrame, optional): CO2 concentration data
    
    Returns:
        tuple: (success, param_dict, results_df, optimization_info)
    """
    try:
        # Load data based on step
        if step == "step1":
            data_file = "data/input/Data_regression_piControl.csv"
            filtered_df = load_and_filter_data(data_file, region, model)
        elif step == "step2":
            filtered_df = load_and_filter_data_step2(region, model)
        elif step == "step3":
            filtered_df = load_and_filter_data_step3(region, model)
        elif step == "step4":
            # Step 4 uses the same data as Step 2 (CO2 fertilization data) for validation
            filtered_df = load_and_filter_data_step2(region, model)
        else:
            print(f"ERROR: Unknown step '{step}'")
            return False, {}, pd.DataFrame(), {}
        
        if filtered_df.empty:
            print(f"No data found for {region} / {model}")
            return False, {}, pd.DataFrame(), {}
        
        # If no parameters to optimize, just run simulation with fixed parameters
        if not params_to_optimize:
            print(f"Running simulation with fixed parameters for {region} / {model}")
            
            # Add derived parameters
            complete_params = fixed_params.copy()
            alpha = complete_params.get('alpha', 0.5)
            Ksoil = complete_params.get('Ksoil_0', 0.1)
            complete_params['Cland_init'] = first_guess_user_params(filtered_df, alpha, Ksoil)
            
            # Run simulation
            results_df = run_bgc_simulation(filtered_df, complete_params, co2_df)
            
            # Add metadata
            complete_params.update({
                'region': region,
                'model': model,
                'step': step,
                'optimization_success': True,
                'final_mse': 0.0  # No optimization performed
            })
            
            optimization_info = {
                'success': True,
                'iterations': 0,
                'function_evaluations': 0,
                'initial_mse': 0.0,
                'final_mse': 0.0,
                'optimized_parameters': [],
                'fixed_parameters': list(fixed_params.keys())
            }
            
            return True, complete_params, results_df, optimization_info
        
        # Run optimization
        print(f"Running optimization for {region} / {model} (Step {step})")
        success, optimized_params, results_df, optimization_info = optimize_parameters(
            fixed_params, params_to_optimize, filtered_df, co2_df
        )
        
        if success:
            # Add metadata
            optimized_params.update({
                'region': region,
                'model': model,
                'step': step,
                'optimization_success': True,
                'final_mse': optimization_info['final_mse']
            })
        
        return success, optimized_params, results_df, optimization_info
        
    except Exception as e:
        print(f"Error processing {region} / {model}: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, pd.DataFrame(), {'success': False, 'error': str(e)}
