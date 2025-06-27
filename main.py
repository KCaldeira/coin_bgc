import pandas as pd

def load_and_filter_data(filepath, region, model):
    # Columns to keep
    columns = ['year', 'tas', 'pr', 'gpp', 'npp', 'region', 'model']
    # Load CSV
    df = pd.read_csv(filepath, usecols=columns)
    # Filter for region and model
    filtered = df[(df['region'] == region) & (df['model'] == model)]
    return filtered

def build_params(filtered_df, user_params):
    # Compute Cland_init from NPP average and Ksoil
    # averaging period is the first 50 years of the model / country combination
    avg_start = filtered_df['year'].min()
    avg_end = avg_start + 50

    # compute Cland_init from NPP average and Ksoil
    Ksoil = user_params['Ksoil'] # inverse years
    avg_npp = filtered_df[(filtered_df['year'] >= avg_start) & (filtered_df['year'] <= avg_end)]['npp'].mean()
    Cland_init = avg_npp / Ksoil
    
    # comput Ktfp from Cland, alpha, and GPP
    alpha = user_params['alpha']
    avg_gpp = filtered_df[(filtered_df['year'] >= avg_start) & (filtered_df['year'] <= avg_end)]['gpp'].mean()
    Ktfp = avg_gpp / (Cland_init ** alpha)

    # compute Kresp as the 1 - avg_npp / avg_gpp
    Kresp = 1 - avg_npp / avg_gpp

    params = {}
    params['Ksoil'] = Ksoil
    params['Ktfp'] = Ktfp
    params['Kresp'] = Kresp
    params['alpha'] = user_params['alpha']
    params['Cland_init'] = Cland_init

    # print params
    print(params)
    
    return params

def run_bgc_simulation(filtered_df, params):
    # Sort by year
    filtered_df = filtered_df.sort_values('year').reset_index(drop=True)
    # print size of filtered_df along with name of region and model
    print(f"Size of filtered_df: {filtered_df.shape}")
    print(f"Region: {filtered_df['region'].iloc[0]}")
    print(f"Model: {filtered_df['model'].iloc[0]}")
    years = filtered_df['year'].values
    Ksoil = params['Ksoil']  # inverse years of land carbon biomass lifetime
    alpha = params['alpha']  # return on investment in carbon biomass
    Cland = params['Cland_init']
    Ktfp = params['Ktfp']
    Kresp = params['Kresp']
    # Store results
    results = []
    for i, row in filtered_df.iterrows():
        year = row['year']
        GPP = Ktfp * (Cland ** alpha)
        NPP = (1 - Kresp) * GPP
        Sresp = Ksoil * Cland
        dCland_dt = NPP - Sresp
        results.append({
            'year': year,
            'Cland': Cland,
            'GPP': GPP,
            'NPP': NPP,
            'Sresp': Sresp,
            'dCland_dt': dCland_dt,
            'tas_data': row['tas'],
            'pr_data': row['pr'],
            'npp_data': row['npp'],
            'gpp_data': row['gpp'],
            'region': row['region'],
            'model': row['model']
        })
        # March forward
        Cland = Cland + dCland_dt  # dt = 1 year
    results_df = pd.DataFrame(results)
    print(results_df.head())
    return results_df

if __name__ == "__main__":
    filepath = "data/input/Data_regression_historical.csv"
    region = "Zimbabwe"
    model = "ACCESS-ESM1-5"
    filtered_df = load_and_filter_data(filepath, region, model)
    # User-supplied parameters
    user_params = {
        'Ksoil': 0.02, # inverse years
        'alpha': 0.3 # dimensionless exponent for power law scaleing of production with Cland
    }
    params = build_params(filtered_df, user_params)
    results_df = run_bgc_simulation(filtered_df, params) 
    # save results_df to csv
    results_df.to_csv(f"data/output/{region}_{model}_results.csv", index=False)