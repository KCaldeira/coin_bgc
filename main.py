import pandas as pd

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
        Ktfp  = params['Ktfp_0']  + params['Ktfp_tas']  * tas + params['Ktfp_pr']  * pr
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
    """Return a dictionary of first-guess user parameters for the model."""
    avg_start = filtered_df['year'].min()
    avg_end = avg_start + n_years - 1

    # compute Cland_init from NPP average and Ksoil
    avg_npp = filtered_df[(filtered_df['year'] >= avg_start) & (filtered_df['year'] <= avg_end)]['npp'].mean()
    Cland_init = avg_npp / Ksoil
    
    # comput Ktfp from Cland, alpha, and GPP
    avg_gpp = filtered_df[(filtered_df['year'] >= avg_start) & (filtered_df['year'] <= avg_end)]['gpp'].mean()
    Ktfp = avg_gpp / (Cland_init ** alpha)

    # compute Kresp as the 1 - avg_npp / avg_gpp
    Kresp = 1 - avg_npp / avg_gpp

    # print npp_avg and gpp_avg
    print(f"npp_avg: {avg_npp}")
    print(f"gpp_avg: {avg_gpp}")

    return {
        'Ksoil_0': Ksoil,
        'Ksoil_tas': 0.0,
        'Ksoil_pr': 0.0,
        'Kresp_0': Kresp,
        'Kresp_tas': 0.0,
        'Kresp_pr': 0.0,
        'Ktfp_0': Ktfp,
        'Ktfp_tas': 0.0,
        'Ktfp_pr': 0.0,
        'alpha': alpha,
        'Cland_init': Cland_init
    }

if __name__ == "__main__":
    filepath = "data/input/Data_regression_historical.csv"
    region = "Zimbabwe"
    model = "ACCESS-ESM1-5"
    filtered_df = load_and_filter_data(filepath, region, model)
    # Get user-supplied parameters from a function
    params = first_guess_user_params(filtered_df, 50, 0.3, 0.05) # 50 years of averaging, alpha = 0.3, Ksoil = 0.05 inverse years

    results_df = run_bgc_simulation(filtered_df, params)
    # save results_df to csv
    results_df.to_csv(f"data/output/{region}_{model}_results.csv", index=False)