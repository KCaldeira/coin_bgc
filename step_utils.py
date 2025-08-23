import pandas as pd
import numpy as np
import os
from datetime import datetime
import scipy.optimize as optimize
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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
    """
    Load and filter data for a specific region and model.
    """
    # Columns to keep
    columns = ['year', 'tas', 'pr', 'gpp', 'npp', 'region', 'model']
    # Load CSV
    df = pd.read_csv(filepath, usecols=columns)
    # Filter for region and model
    filtered = df[(df['region'] == region) & (df['model'] == model)]
    return filtered

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
    
    years = filtered_df['year'].values
    alpha = params.get('alpha', 0.5)  # dimensionless exponent for power law scaling of production with Cland
    Cland = params.get('Cland_init', 1.0)
    
    # Store results
    results = []
    for i, row in filtered_df.iterrows():
        year = row['year']
        tas = row['tas']
        pr = row['pr']
        
        # Get CO2 concentration for this year
        co2 = get_co2_for_year(co2_df, year) if co2_df is not None else co2_0
        
        # Calculate Ksoil, Kresp, Ktfp as linear functions of tas and pr
        Ksoil = params.get('Ksoil_0', 0.1) + params.get('Ksoil_tas', 0.0) * tas + params.get('Ksoil_pr', 0.0) * pr
        Kresp = params.get('Kresp_0', 0.5) + params.get('Kresp_tas', 0.0) * tas + params.get('Kresp_pr', 0.0) * pr
        
        # Calculate Ktfp with optional CO2 dependence
        Ktfp_base = params.get('Ktfp_0', 1.0) * (1 + params.get('Ktfp_tas', 0.0) * tas + params.get('Ktfp_pr', 0.0) * pr)
        
        if 'Ktfp_co2' in params and co2_df is not None:
            # CO2-dependent Ktfp calculation
            Ktfp = Ktfp_base * (1 + params['Ktfp_co2']) * ((co2/co2_0) / (params['Ktfp_co2'] + co2/co2_0))
        else:
            Ktfp = Ktfp_base
        
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
    results_df = run_bgc_simulation(filtered_df, param_dict)
    
    # Calculate error (MSE between simulated and observed NPP)
    mse = np.mean((results_df['NPP'] - results_df['npp_data'])**2)
    return mse

def run_single_region_model(region, model, args, user_params, co2_df=None):
    """
    Run simulation for a single region/model combination.
    """
    try:
        # Load data
        if args.step == "step1":
            data_file = "data/input/Data_regression_piControl.csv"
        elif args.step == "step2":
            data_file = "data/input/Data_regression_ssp585-bgc.csv"
        elif args.step == "step3":
            data_file = "data/input/Data_regression_ssp585.csv"
        else:
            data_file = "data/input/Data_regression_piControl.csv"  # default
        
        filtered_df = load_and_filter_data(data_file, region, model)
        
        if filtered_df.empty:
            print(f"No data found for {region} / {model}")
            return False, {}, pd.DataFrame()
        
        # Check if user provided all main parameters AND we're not in step2 (where we want to optimize Ktfp_co2)
        main_params = ['Ksoil_0', 'Kresp_0', 'Ktfp_0', 'alpha']
        provided_main_params = [p for p in main_params if p in user_params and user_params[p] is not None]
        
        # For step2, we want to optimize Ktfp_co2 even if all main parameters are provided
        # For step3, we want to optimize climate sensitivity parameters even if all main parameters are provided
        if len(provided_main_params) == len(main_params) and args.step not in ["step2", "step3"]:
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
            param_bounds.append((0.01, 0.5))
            initial_guess.append(0.1)
        
        if 'Kresp_0' not in user_params:
            param_names.append('Kresp_0')
            param_bounds.append((0.1, 0.9))
            initial_guess.append(0.5)
        
        if 'Ktfp_0' not in user_params:
            param_names.append('Ktfp_0')
            param_bounds.append((0.1, 10.0))
            initial_guess.append(1.0)
        
        if 'alpha' not in user_params:
            param_names.append('alpha')
            param_bounds.append((0.1, 1.0))
            initial_guess.append(0.5)
        
        # Add CO2 parameter for step2
        if args.step == "step2" and 'Ktfp_co2' not in user_params:
            param_names.append('Ktfp_co2')
            param_bounds.append((0.0, 2.0))
            initial_guess.append(0.1)
        
        # Add climate sensitivity parameters for step3
        if args.step == "step3":
            climate_params = ['Ksoil_tas', 'Ksoil_pr', 'Kresp_tas', 'Kresp_pr', 'Ktfp_tas', 'Ktfp_pr']
            for param in climate_params:
                # For step3, always optimize climate sensitivity parameters
                param_names.append(param)
                if 'tas' in param:
                    param_bounds.append((-0.1, 0.1))  # Temperature sensitivity bounds
                    initial_guess.append(user_params.get(param, 0.0))
                else:
                    param_bounds.append((-0.01, 0.01))  # Precipitation sensitivity bounds
                    initial_guess.append(user_params.get(param, 0.0))
        
        if not param_names:
            print("No parameters to optimize")
            return False, {}, pd.DataFrame()
        
        # Run optimization
        result = optimize.minimize(
            objective_function,
            initial_guess,
            args=(filtered_df, param_names, user_params),
            bounds=param_bounds,
            method='L-BFGS-B'
        )
        
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

def save_fitted_parameters(all_fitted_params, step, single_file=True):
    """
    Save fitted parameters to CSV file(s).
    """
    if not all_fitted_params:
        return
    
    output_dir = setup_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if single_file:
        # Save all parameters to one file
        all_params_df = pd.DataFrame(all_fitted_params)
        filename = f"fitted_parameters_all_{step}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        all_params_df.to_csv(filepath, index=False)
        print(f"All fitted parameters saved to: {filepath}")
    else:
        # Save individual files (legacy option)
        for i, params in enumerate(all_fitted_params):
            region = params.get('region', f'region_{i}')
            model = params.get('model', f'model_{i}')
            filename = f"fitted_parameters_{region}_{model}_{step}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            pd.DataFrame([params]).to_csv(filepath, index=False)

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
