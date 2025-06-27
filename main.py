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
    years = filtered_df['year'].values
    # Averaging period for initial Cland
    avg_start = params['avg_start_year']
    avg_end = params['avg_end_year']
    Ksoil = params['Ksoil']
    Ktfp = params['Ktfp']
    Kresp = params['Kresp']
    alpha = params['alpha']
    # Average NPP over the period
    avg_npp = filtered_df[(filtered_df['year'] >= avg_start) & (filtered_df['year'] <= avg_end)]['npp'].mean()
    Cland = avg_npp / Ksoil
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
            'dCland_dt': dCland_dt
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
    # Example parameters
    params = {
        'Ksoil': 0.05,
        'Ktfp': 1.0,
        'Kresp': 0.4,
        'alpha': 1.0,
        'avg_start_year': 1850,
        'avg_end_year': 1900
    }
    run_bgc_simulation(filtered_df, params) 