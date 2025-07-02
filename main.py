import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.optimize as optimize

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

def step1_picontrol_parameter_estimation(region, model, n_years=1):
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
        print(f"Available regions: {full_df['region'].unique()}")
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
    
    # Define objective function for optimization
    def objective_function(params):
        Ksoil_0, Kresp_0, Ktfp_0, alpha = params
        
        # Set up parameter dictionary with _tas and _pr coefficients as zero
        user_params = {
            'Ksoil_0': Ksoil_0,
            'Ksoil_tas': 0.0,
            'Ksoil_pr': 0.0,
            'Kresp_0': Kresp_0,
            'Kresp_tas': 0.0,
            'Kresp_pr': 0.0,
            'Ktfp_0': Ktfp_0,
            'Ktfp_tas': 0.0,
            'Ktfp_pr': 0.0,
            'alpha': alpha
        }
        
        # Build full params dictionary
        params_dict = build_params(filtered_df, user_params)
        
        # Run simulation
        results_df = run_bgc_simulation(filtered_df, params_dict)
        
        # Calculate error (difference between predicted and observed GPP/NPP)
        gpp_error = np.mean((results_df['GPP'] - results_df['gpp_data'])**2)
        npp_error = np.mean((results_df['NPP'] - results_df['npp_data'])**2)
        
        total_error = gpp_error + npp_error
        return total_error
    
    # Initial guess for parameters
    initial_guess = [0.05, 0.4, 1.0, 0.3]  # [Ksoil_0, Kresp_0, Ktfp_0, alpha]
    
    # Bounds for parameters (all positive, alpha typically between 0.1 and 0.8)
    bounds = [(0.001, 1.0), (0.001, 0.99), (0.001, 10.0), (0.1, 0.8)]
    
    # Optimize
    result = optimize.minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        Ksoil_0_opt, Kresp_0_opt, Ktfp_0_opt, alpha_opt = result.x
        # Estimate standard errors from Hessian inverse
        try:
            # For L-BFGS-B, hess_inv is a LinearOperator; convert to dense
            hess_inv = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv
            param_std = np.sqrt(np.diag(hess_inv))
            print(f"Standard errors: Ksoil_0={param_std[0]:.4g}, Kresp_0={param_std[1]:.4g}, Ktfp_0={param_std[2]:.4g}, alpha={param_std[3]:.4g}")
        except Exception as e:
            print(f"Could not compute standard errors: {e}")
            param_std = [None, None, None, None]
        print(f"Optimization successful!")
        print(f"Optimal parameters: Ksoil_0={Ksoil_0_opt:.4f}, Kresp_0={Kresp_0_opt:.4f}, Ktfp_0={Ktfp_0_opt:.4f}, alpha={alpha_opt:.4f}")
        print(f"Final error: {result.fun:.6f}")
        
        # Return optimal parameters (including standard errors)
        optimal_params = {
            'Ksoil_0': Ksoil_0_opt,
            'Ksoil_0_stderr': param_std[0],
            'Kresp_0': Kresp_0_opt,
            'Kresp_0_stderr': param_std[1],
            'Ktfp_0': Ktfp_0_opt,
            'Ktfp_0_stderr': param_std[2],
            'alpha': alpha_opt,
            'alpha_stderr': param_std[3],
            'Ksoil_tas': 0.0,
            'Ksoil_pr': 0.0,
            'Kresp_tas': 0.0,
            'Kresp_pr': 0.0,
            'Ktfp_tas': 0.0,
            'Ktfp_pr': 0.0
        }
        
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
    # If Cland_init is not provided, compute it from steady-state condition
    if 'Cland_init' not in params or params['Cland_init'] is None:
        avg_npp = filtered_df['npp'].mean()
        Ksoil = params.get('Ksoil_0', 0.04)
        params['Cland_init'] = avg_npp / Ksoil if Ksoil != 0 else 1.0
    return params

def save_fitted_parameters(fitted_params_list, output_file="data/output/step1_fitted_parameters.csv"):
    """
    Save fitted parameters for all region/model combinations to a CSV file.
    Each row represents one region/model combination with all its parameters and standard errors.
    """
    # Convert list of parameter dictionaries to DataFrame
    params_df = pd.DataFrame(fitted_params_list)
    
    # Keep only the base parameters and their standard errors for Step 1
    base_params = [
        'region', 'model',
        'Ksoil_0', 'Ksoil_0_stderr',
        'Kresp_0', 'Kresp_0_stderr',
        'Ktfp_0', 'Ktfp_0_stderr',
        'alpha', 'alpha_stderr'
    ]
    params_df = params_df[base_params]
    
    # Save to CSV
    params_df.to_csv(output_file, index=False)
    print(f"Fitted parameters saved to {output_file}")
    return params_df

if __name__ == "__main__":
    region = "Zimbabwe"
    model = "ACCESS-ESM1-5"
    
    # Step 1: Estimate base parameters using piControl data
    optimal_params, filtered_df = step1_picontrol_parameter_estimation(region, model)
    
    if optimal_params is not None:
        # Add region and model info to parameters
        optimal_params['region'] = region
        optimal_params['model'] = model
        
        # Run simulation with optimal parameters
        params_dict = build_params(filtered_df, optimal_params)
        results_df = run_bgc_simulation(filtered_df, params_dict)
        
        # Save results
        results_df.to_csv(f"data/output/{region}_{model}_step1_results.csv", index=False)
        print(f"Results saved to data/output/{region}_{model}_step1_results.csv")
        
        # Save fitted parameters
        save_fitted_parameters([optimal_params])
    else:
        print("Step 1 failed - no results to save")