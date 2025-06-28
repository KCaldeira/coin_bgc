import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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
    print(f"Size of filtered_df: {filtered_df.shape}")
    print(f"Region: {filtered_df['region'].iloc[0]}")
    print(f"Model: {filtered_df['model'].iloc[0]}")
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
            'npp_data': row['npp'],
            'gpp_data': row['gpp'],
            'region': row['region'],
            'model': row['model'],
            'Ksoil': Ksoil,
            'Kresp': Kresp,
            'Ktfp': Ktfp
        })
        # March forward
        Cland = Cland + dCland_dt  # dt = 1 year
    results_df = pd.DataFrame(results)
    print(results_df.head())
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

if __name__ == "__main__":
    filepath = "data/input/Data_regression_historical.csv"
    region = "Zimbabwe"
    model = "ACCESS-ESM1-5"
    filtered_df = load_and_filter_data(filepath, region, model)
    # Get user-supplied parameters from a function
    params = first_guess_user_params(filtered_df, 25, 0.3, 0.04) # 25 years of averaging, alpha = 0.3, Ksoil = 0.04 inverse years

    results_df = run_bgc_simulation(filtered_df, params)
    # save results_df to csv
    results_df.to_csv(f"data/output/{region}_{model}_results.csv", index=False)