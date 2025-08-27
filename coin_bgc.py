"""
COIN-BGC: Clean Architecture Implementation

This module implements the clean architecture for the COIN-BGC system based on
three fundamental lists of keys: knowns, unknowns, and universe.

Core Design Principles:
- More general: Basic routines are as general as possible
- More specific: Written for one use case but easily adaptable
- No conditional clutter: Avoid complex conditional statements

Central Design: Three Lists of Keys
1. knowns - Variables to be specified in optimization (user-provided parameters)
2. unknowns - Variables to be optimized for (parameters to be determined)  
3. universe - Complete set of variables (all possible parameters)
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional


def load_data_for_analysis(regions: Optional[List[str]] = None, models: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three main datasets for COIN-BGC analysis.
    
    Args:
        regions: Optional list of geographic regions to filter by (e.g., ["Zimbabwe", "China"])
        models: Optional list of climate models to filter by (e.g., ["ACCESS-ESM1-5"])
        
    Returns:
        Tuple of (piControl_data, full_data, bgc_data)
        
        piControl_data: Pre-industrial control data for Step 1
        full_data: Historical + SSP585 data for Step 3 (both CO2 and climate change)
        bgc_data: Historical-bgc + SSP585-bgc data for Step 2 (CO2 changes only)
    """
    # Columns to keep
    columns = ['year', 'tas', 'pr', 'gpp', 'npp', 'region', 'model']
    
    # Load piControl data (pre-industrial control)
    piControl_file = "data/input/Data_regression_piControl.csv"
    piControl_df = pd.read_csv(piControl_file, usecols=columns)
    piControl_data = piControl_df.copy()
    
    # Apply filters if specified
    if regions is not None:
        piControl_data = piControl_data[piControl_data['region'].isin(regions)]
    if models is not None:
        piControl_data = piControl_data[piControl_data['model'].isin(models)]
    
    piControl_data = piControl_data.sort_values('year').reset_index(drop=True)
    
    # Load historical data
    hist_file = "data/input/Data_regression_historical.csv"
    hist_df = pd.read_csv(hist_file, usecols=columns)
    hist_filtered = hist_df.copy()
    
    # Apply filters if specified
    if regions is not None:
        hist_filtered = hist_filtered[hist_filtered['region'].isin(regions)]
    if models is not None:
        hist_filtered = hist_filtered[hist_filtered['model'].isin(models)]
    
    # Load SSP585 data (both CO2 and climate change)
    ssp585_file = "data/input/Data_regression_ssp585.csv"
    ssp585_df = pd.read_csv(ssp585_file, usecols=columns)
    ssp585_filtered = ssp585_df.copy()
    
    # Apply filters if specified
    if regions is not None:
        ssp585_filtered = ssp585_filtered[ssp585_filtered['region'].isin(regions)]
    if models is not None:
        ssp585_filtered = ssp585_filtered[ssp585_filtered['model'].isin(models)]
    
    # Create full_data (historical + SSP585)
    full_data = pd.concat([hist_filtered, ssp585_filtered], ignore_index=True)
    full_data = full_data.sort_values('year').reset_index(drop=True)
    
    # Load historical-bgc data
    hist_bgc_file = "data/input/Data_regression_hist-bgc.csv"
    hist_bgc_df = pd.read_csv(hist_bgc_file, usecols=columns)
    hist_bgc_filtered = hist_bgc_df.copy()
    
    # Apply filters if specified
    if regions is not None:
        hist_bgc_filtered = hist_bgc_filtered[hist_bgc_filtered['region'].isin(regions)]
    if models is not None:
        hist_bgc_filtered = hist_bgc_filtered[hist_bgc_filtered['model'].isin(models)]
    
    # Load SSP585-bgc data (CO2 changes only)
    ssp585_bgc_file = "data/input/Data_regression_ssp585-bgc.csv"
    ssp585_bgc_df = pd.read_csv(ssp585_bgc_file, usecols=columns)
    ssp585_bgc_filtered = ssp585_bgc_df.copy()
    
    # Apply filters if specified
    if regions is not None:
        ssp585_bgc_filtered = ssp585_bgc_filtered[ssp585_bgc_filtered['region'].isin(regions)]
    if models is not None:
        ssp585_bgc_filtered = ssp585_bgc_filtered[ssp585_bgc_filtered['model'].isin(models)]
    
    # Create bgc_data (historical-bgc + SSP585-bgc)
    bgc_data = pd.concat([hist_bgc_filtered, ssp585_bgc_filtered], ignore_index=True)
    bgc_data = bgc_data.sort_values('year').reset_index(drop=True)
    
    return piControl_data, full_data, bgc_data


def load_co2_data() -> pd.DataFrame:
    """
    Load CO2 concentration data from the historical-SSP585 file.
    
    Returns:
        DataFrame with 'year' and 'co2' columns
    """
    co2_file = "data/input/historical-ssp585_co2.csv"
    co2_df = pd.read_csv(co2_file)
    # Rename column for consistency
    co2_df = co2_df.rename(columns={'pco2 (ppm)': 'co2'})
    return co2_df


class CoinBGC:
    """
    COIN-BGC: Clean Architecture Implementation
    
    This class implements the clean architecture with three lists of keys
    and two core routines for model execution and optimization.
    """
    
    def __init__(self):
        """Initialize the COIN-BGC system."""
        # Define the complete universe of variables
        self.universe = [
            'Ksoil_0',      # Inverse time constant for soil respiration
            'Kresp_0',      # Plant respiration fraction
            'Ktfp_0',       # Total factor productivity (base)
            'alpha',        # Production function exponent
            'Cland_0',      # Initial carbon land stock
            'Ktfp_co2',     # CO2 fertilization sensitivity
            'Ktfp_co2_max', # CO2 fertilization maximum factor
            'Ktfp_tas0',    # Reference temperature
            'Ktfp_tas1',    # Temperature sensitivity coefficient (linear)
            'Ktfp_tas2',    # Temperature sensitivity coefficient (quadratic)
            'Ktfp_pr0',     # Reference precipitation
            'Ktfp_pr1',     # Precipitation sensitivity coefficient (linear)
            'Ktfp_pr2'      # Precipitation sensitivity coefficient (quadratic)
        ]
        
        # Reference CO2 concentration (pre-industrial)
        self.co2_0 = 284.317  # ppm
        
    def set_parameter_sets(self, knowns: List[str], unknowns: List[str]) -> None:
        """
        Set the knowns and unknowns parameter sets.
        
        Args:
            knowns: List of parameter names that are known/specified
            unknowns: List of parameter names to be optimized
        """
        self.knowns = knowns
        self.unknowns = unknowns
        
        # Validate that knowns and unknowns are subsets of universe
        universe_set = set(self.universe)
        knowns_set = set(knowns)
        unknowns_set = set(unknowns)
        
        if not knowns_set.issubset(universe_set):
            raise ValueError(f"Unknown parameters in knowns: {knowns_set - universe_set}")
        
        if not unknowns_set.issubset(universe_set):
            raise ValueError(f"Unknown parameters in unknowns: {unknowns_set - universe_set}")
        
        # Check for overlap between knowns and unknowns
        overlap = knowns_set.intersection(unknowns_set)
        if overlap:
            raise ValueError(f"Parameters cannot be both known and unknown: {overlap}")
    
    def create_parameter_dict(self, known_values: Dict[str, float], 
                            unknown_values: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Create a complete parameter dictionary from known and unknown values.
        
        Args:
            known_values: Dictionary of known parameter values
            unknown_values: Dictionary of unknown parameter values (optional)
            
        Returns:
            Complete parameter dictionary with all universe parameters
        """
        # Start with all parameters set to 0.0 (universe - knowns)
        params = {param: 0.0 for param in self.universe}
        
        # Set known parameter values
        params.update(known_values)
        
        # Set unknown parameter values if provided
        if unknown_values:
            params.update(unknown_values)
            
        return params
    
    def calculate_ktfp(self, params: Dict[str, float], tas: float, pr: float, co2: float) -> float:
        """
        Calculate total factor productivity (Ktfp) based on climate and CO2.
        
        Args:
            params: Complete parameter dictionary
            tas: Temperature (°C)
            pr: Precipitation (mm/day)
            co2: CO2 concentration (ppm)
            
        Returns:
            Calculated Ktfp value
        """
        ktfp_0 = params['Ktfp_0']
        
        # Temperature factor
        tas_factor = 1.0 + params['Ktfp_tas1'] * (tas - params['Ktfp_tas0']) + params['Ktfp_tas2'] * (tas - params['Ktfp_tas0'])**2
        
        # Precipitation factor  
        pr_factor = 1.0 + params['Ktfp_pr1'] * (pr - params['Ktfp_pr0']) + params['Ktfp_pr2'] * (pr - params['Ktfp_pr0'])**2
        
        # CO2 factor
        co2_factor = 1.0 + params['Ktfp_co2_max'] * (co2 - self.co2_0) / params['Ktfp_co2']
        
        return ktfp_0 * tas_factor * pr_factor * co2_factor
    
    def execute_model(self, data_df: pd.DataFrame, known_values: Dict[str, float], 
                     co2_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Basic Model Execution Routine: Execute the model from start year to end year.
        
        This is the first core routine that takes a pandas DataFrame and a dictionary
        of parameter values for the set of knowns. All variables not in knowns
        (i.e., universe - knowns) are set to 0.0.
        
        Args:
            data_df: DataFrame with columns ['year', 'tas', 'pr', 'npp', 'gpp']
            known_values: Dictionary of known parameter values
            co2_df: Optional DataFrame with CO2 concentration data
            
        Returns:
            DataFrame with model results including calculated GPP, NPP, Cland, etc.
        """
        # Create complete parameter dictionary (unknowns set to 0.0)
        params = self.create_parameter_dict(known_values)
        
        # Sort data by year
        data_df = data_df.sort_values('year').reset_index(drop=True)
        

        
        # Get parameters (assume all are provided)
        alpha = params['alpha']
        Ksoil = params['Ksoil_0']
        Kresp = params['Kresp_0']
        Ktfp_0 = params['Ktfp_0']
        Cland_0 = params['Cland_0']
        
        # Initialize Cland
        Cland = Cland_0
        
        # Store results
        results = []
        
        # Run model year by year
        for i, row in data_df.iterrows():
            year = row['year']
            tas = row['tas']
            pr = row['pr']
            
            # Get CO2 concentration for this year
            co2 = self._get_co2_for_year(co2_df, year) if co2_df is not None else self.co2_0
            
            # Calculate Ktfp for this year
            Ktfp = self.calculate_ktfp(params, tas, pr, co2)
            
            # Calculate GPP
            GPP = Ktfp * (Cland ** alpha)
            
            # Calculate NPP
            NPP = (1 - Kresp) * GPP
            
            # Calculate soil respiration
            Sresp = Ksoil * Cland
            
            # Calculate change in Cland
            dCland_dt = NPP - Sresp
            
            # Store results for this year
            results.append({
                'year': year,
                'Cland': Cland,
                'GPP_model': GPP,
                'NPP_model': NPP,
                'SOILresp': Sresp,
                'dCland_dt': dCland_dt,
                'tas_data': tas,
                'pr_data': pr,
                'co2': co2,
                'gpp_data': row['gpp'],
                'npp_data': row['npp'],
                'region': row.get('region', 'unknown'),
                'model': row.get('model', 'unknown'),
                'Ksoil': Ksoil,
                'Kresp': Kresp,
                'Ktfp': Ktfp
            })
            
            # Update Cland for next year
            Cland = Cland + dCland_dt
        
        return pd.DataFrame(results)
    
    def objective_function(self, unknown_values: np.ndarray, known_values: Dict[str, float], 
                          data_df: pd.DataFrame, co2_df: Optional[pd.DataFrame] = None) -> float:
        """
        Objective function for optimization: Mean squared error between observed and predicted GPP.
        
        Args:
            unknown_values: Array of unknown parameter values
            known_values: Dictionary of known parameter values
            data_df: DataFrame with observed data
            co2_df: Optional DataFrame with CO2 concentration data
            
        Returns:
            Mean squared error
        """
        # Convert unknown_values array to dictionary
        unknown_dict = dict(zip(self.unknowns, unknown_values))
        
        # Create complete parameter dictionary
        params = self.create_parameter_dict(known_values, unknown_dict)
        
        # Run model
        results = self.execute_model(data_df, known_values, co2_df)
        
        # Calculate MSE between observed and predicted GPP
        mse = np.mean((results['gpp_data'] - results['GPP_model']) ** 2)
        
        return mse
    
    def optimize_parameters(self, known_values: Dict[str, float], data_df: pd.DataFrame,
                          initial_guesses: Optional[Dict[str, float]] = None,
                          bounds: Optional[List[Tuple[float, float]]] = None,
                          co2_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Optimization Routine: Optimize for the unknowns given the knowns.
        
        This is the second core routine that optimizes for the unknowns given the knowns
        and a pandas DataFrame. Returns optimal values for the unknown parameters.
        
        Args:
            known_values: Dictionary of known parameter values
            data_df: DataFrame with observed data
            initial_guesses: Dictionary of initial guesses for unknown parameters
            bounds: List of (min, max) bounds for unknown parameters
            co2_df: Optional DataFrame with CO2 concentration data
            
        Returns:
            Dictionary of optimal values for unknown parameters
        """
        # Create initial guess array
        x0 = np.array([initial_guesses[param] for param in self.unknowns])
        
        # Create bounds array
        bounds_array = [bounds[i] for i, param in enumerate(self.unknowns)]
        
        # Run optimization
        result = minimize(
            fun=self.objective_function,
            x0=x0,
            args=(known_values, data_df, co2_df),
            bounds=bounds_array,
            method='L-BFGS-B'
        )
        

        
        # Convert result to dictionary
        optimal_unknowns = dict(zip(self.unknowns, result.x))
        
        return optimal_unknowns
    
    def objective_function_multi(self, unknown_values: np.ndarray, known_values: Dict[str, float], 
                                data_dfs: List[pd.DataFrame], co2_dfs: Optional[List[pd.DataFrame]] = None) -> float:
        """
        Objective function for multi-DataFrame optimization: Combined MSE across all DataFrames.
        
        Args:
            unknown_values: Array of unknown parameter values
            known_values: Dictionary of known parameter values
            data_dfs: List of DataFrames with observed data
            co2_dfs: Optional list of DataFrames with CO2 concentration data
            
        Returns:
            Combined mean squared error across all DataFrames
        """
        # Convert unknown_values array to dictionary
        unknown_dict = dict(zip(self.unknowns, unknown_values))
        
        # Create complete parameter dictionary
        params = self.create_parameter_dict(known_values, unknown_dict)
        
        total_mse = 0.0
        total_points = 0
        
        # Run model for each DataFrame and accumulate MSE
        for i, data_df in enumerate(data_dfs):
            # Get corresponding CO2 DataFrame
            co2_df = co2_dfs[i]
            
            # Run model for this DataFrame
            results = self.execute_model(data_df, known_values, co2_df)
            
            # Calculate MSE for this DataFrame
            mse = np.mean((results['gpp_data'] - results['GPP_model']) ** 2)
            n_points = len(results)
            
            # Accumulate weighted MSE
            total_mse += mse * n_points
            total_points += n_points
        
        # Return average MSE across all DataFrames
        return total_mse / total_points
    
    def optimize_parameters_multi(self, known_values: Dict[str, float], data_dfs: List[pd.DataFrame],
                                initial_guesses: Optional[Dict[str, float]] = None,
                                bounds: Optional[List[Tuple[float, float]]] = None,
                                co2_dfs: Optional[List[pd.DataFrame]] = None) -> Dict[str, float]:
        """
        Multi-DataFrame Optimization Routine: Optimize for the unknowns given the knowns.
        
        This variant of the optimization routine optimizes for the unknowns given the knowns
        and a list of pandas DataFrames. The model is run individually for each DataFrame
        using the same set of parameters, and the objective function minimizes the combined
        MSE across all simulations.
        
        Args:
            known_values: Dictionary of known parameter values
            data_dfs: List of DataFrames with observed data
            initial_guesses: Dictionary of initial guesses for unknown parameters
            bounds: List of (min, max) bounds for unknown parameters
            co2_dfs: Optional list of DataFrames with CO2 concentration data
            
        Returns:
            Dictionary of optimal values for unknown parameters
        """
        # Create initial guess array
        x0 = np.array([initial_guesses[param] for param in self.unknowns])
        
        # Create bounds array
        bounds_array = [bounds[i] for i, param in enumerate(self.unknowns)]
        
        # Run optimization
        result = minimize(
            fun=self.objective_function_multi,
            x0=x0,
            args=(known_values, data_dfs, co2_dfs),
            bounds=bounds_array,
            method='L-BFGS-B'
        )
        

        
        # Convert result to dictionary
        optimal_unknowns = dict(zip(self.unknowns, result.x))
        
        return optimal_unknowns
    

    

    
    def _get_co2_for_year(self, co2_df: Optional[pd.DataFrame], year: int) -> float:
        """
        Get CO2 concentration for a specific year from CO2 DataFrame.
        
        Args:
            co2_df: DataFrame with CO2 concentration data
            year: Year to get CO2 for
            
        Returns:
            CO2 concentration for the specified year
        """

        
        # Find the closest year in the CO2 data
        year_diff = abs(co2_df['year'] - year)
        closest_idx = year_diff.idxmin()
        
        return co2_df.loc[closest_idx, 'co2']


def run_preliminary_optimizations(piControl_data: pd.DataFrame, full_data: pd.DataFrame, bgc_data: pd.DataFrame, 
                                co2_data: pd.DataFrame, Ksoil_0: float, alpha: float) -> Dict[str, float]:
    """
    Run preliminary optimizations to get starting points for complete optimization.
    
    Args:
        piControl_data: Pre-industrial control data
        full_data: Historical + SSP585 data
        bgc_data: Historical-bgc + SSP585-bgc data
        co2_data: CO2 concentration data
        Ksoil_0: Soil respiration parameter (from command line)
        alpha: Production function exponent (from command line)
        
    Returns:
        Dictionary of optimized parameters
    """
    model = CoinBGC()
    
    # Step 2.1: Calculate Cland_0, Kresp_0, and Ktfp_0 based on mean values from historical_data
    print("=== Step 2.1: Calculating initial parameters from historical data ===")
    
    # Use historical portion of full_data for initial calculations
    historical_data = full_data[full_data['year'] <= 2014].copy()  # Historical period
    
    # Calculate mean values
    gpp_mean = historical_data['gpp'].mean()
    npp_mean = historical_data['npp'].mean()
    tas_mean = historical_data['tas'].mean()
    pr_mean = historical_data['pr'].mean()
    
    # Calculate initial parameters
    Kresp_0 = npp_mean / gpp_mean  # From steady-state: NPP = (1 - Kresp_0) * GPP
    Cland_0 = npp_mean / Ksoil_0   # From steady-state: NPP = Ksoil_0 * Cland_0
    Ktfp_0 = gpp_mean / (Cland_0 ** alpha)  # From production function: GPP = Ktfp_0 * Cland_0^alpha
    
    print(f"Initial parameters from historical data:")
    print(f"  Kresp_0: {Kresp_0:.6f}")
    print(f"  Cland_0: {Cland_0:.6f}")
    print(f"  Ktfp_0: {Ktfp_0:.6f}")
    print(f"  tas_mean: {tas_mean:.6f}")
    print(f"  pr_mean: {pr_mean:.6f}")
    
    # Step 2.2: Set reference values to historical means
    Ktfp_tas0 = tas_mean
    Ktfp_pr0 = pr_mean
    
    # Initialize parameters dictionary
    params = {
        'Ksoil_0': Ksoil_0,
        'alpha': alpha,
        'Kresp_0': Kresp_0,
        'Cland_0': Cland_0,
        'Ktfp_0': Ktfp_0,
        'Ktfp_tas0': Ktfp_tas0,
        'Ktfp_pr0': Ktfp_pr0,
        'Ktfp_tas1': 0.0,
        'Ktfp_tas2': 0.0,
        'Ktfp_pr1': 0.0,
        'Ktfp_pr2': 0.0,
        'Ktfp_co2': 10.0,
        'Ktfp_co2_max': 1.0
    }
    
    # Step 2.3: Optimize Cland_0, Ktfp_0, Ktfp_tas1, Ktfp_tas2, Ktfp_pr1, Ktfp_pr2 using historical data
    print("\n=== Step 2.3: Optimizing climate sensitivity parameters using historical data ===")
    
    # Set knowns and unknowns for this step
    knowns = ['Ksoil_0', 'alpha', 'Kresp_0', 'Ktfp_tas0', 'Ktfp_pr0', 'Ktfp_co2', 'Ktfp_co2_max']
    unknowns = ['Cland_0', 'Ktfp_0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr1', 'Ktfp_pr2']
    
    model.set_parameter_sets(knowns, unknowns)
    
    # Create initial guesses and bounds
    initial_guesses = {
        'Cland_0': Cland_0,
        'Ktfp_0': Ktfp_0,
        'Ktfp_tas1': 0.0,
        'Ktfp_tas2': 0.0,
        'Ktfp_pr1': 0.0,
        'Ktfp_pr2': 0.0
    }
    
    bounds = [
        (Cland_0 * 0.5, Cland_0 * 2.0),      # Cland_0
        (Ktfp_0 * 0.5, Ktfp_0 * 2.0),        # Ktfp_0
        (-0.1, 0.1),                         # Ktfp_tas1
        (-0.01, 0.01),                       # Ktfp_tas2
        (-0.1, 0.1),                         # Ktfp_pr1
        (-0.01, 0.01)                        # Ktfp_pr2
    ]
    
    # Run optimization
    optimal_params = model.optimize_parameters(
        known_values={param: params[param] for param in knowns},
        data_df=historical_data,
        initial_guesses=initial_guesses,
        bounds=bounds
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.3 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Step 2.4: Optimize Ktfp_co2_max and Ktfp_co2 using bgc_data
    print("\n=== Step 2.4: Optimizing CO2 parameters using bgc_data ===")
    
    # Set knowns and unknowns for this step
    knowns = ['Ksoil_0', 'alpha', 'Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2']
    unknowns = ['Ktfp_co2_max', 'Ktfp_co2']
    
    model.set_parameter_sets(knowns, unknowns)
    
    # Create initial guesses and bounds
    initial_guesses = {
        'Ktfp_co2_max': 1.0,
        'Ktfp_co2': 10.0
    }
    
    bounds = [
        (0.5, 2.0),                          # Ktfp_co2_max
        (5.0, 20.0)                          # Ktfp_co2
    ]
    
    # Run optimization
    optimal_params = model.optimize_parameters(
        known_values={param: params[param] for param in knowns},
        data_df=bgc_data,
        initial_guesses=initial_guesses,
        bounds=bounds,
        co2_df=co2_data
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.4 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Step 2.5: Optimize all parameters using bgc_data and piControl_data
    print("\n=== Step 2.5: Optimizing all parameters using bgc_data and piControl_data ===")
    
    # Set knowns and unknowns for this step
    knowns = ['Ksoil_0', 'alpha']  # Only keep command line parameters as knowns
    unknowns = ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_max', 'Ktfp_co2']
    
    model.set_parameter_sets(knowns, unknowns)
    
    # Create initial guesses and bounds
    initial_guesses = {param: params[param] for param in unknowns}
    
    bounds = [
        (0.1, 0.9),                          # Kresp_0
        (params['Cland_0'] * 0.5, params['Cland_0'] * 2.0),  # Cland_0
        (params['Ktfp_0'] * 0.5, params['Ktfp_0'] * 2.0),    # Ktfp_0
        (params['Ktfp_tas0'] * 0.5, params['Ktfp_tas0'] * 2.0),  # Ktfp_tas0
        (-0.1, 0.1),                         # Ktfp_tas1
        (-0.01, 0.01),                       # Ktfp_tas2
        (params['Ktfp_pr0'] * 0.5, params['Ktfp_pr0'] * 2.0),  # Ktfp_pr0
        (-0.1, 0.1),                         # Ktfp_pr1
        (-0.01, 0.01),                       # Ktfp_pr2
        (0.5, 2.0),                          # Ktfp_co2_max
        (5.0, 20.0)                          # Ktfp_co2
    ]
    
    # Run multi-DataFrame optimization
    optimal_params = model.optimize_parameters_multi(
        known_values={param: params[param] for param in knowns},
        data_dfs=[bgc_data, piControl_data],
        initial_guesses=initial_guesses,
        bounds=bounds,
        co2_dfs=[co2_data, None]  # CO2 data for bgc_data, None for piControl_data
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.5 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Step 2.6: Final optimization of climate sensitivity parameters using all data
    print("\n=== Step 2.6: Final optimization of climate sensitivity parameters using all data ===")
    
    # Set knowns and unknowns for this step
    knowns = ['Ksoil_0', 'alpha', 'Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_pr0', 'Ktfp_co2_max', 'Ktfp_co2']
    unknowns = ['Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr1', 'Ktfp_pr2']
    
    model.set_parameter_sets(knowns, unknowns)
    
    # Create initial guesses and bounds
    initial_guesses = {param: params[param] for param in unknowns}
    
    bounds = [
        (-0.1, 0.1),                         # Ktfp_tas1
        (-0.01, 0.01),                       # Ktfp_tas2
        (-0.1, 0.1),                         # Ktfp_pr1
        (-0.01, 0.01)                        # Ktfp_pr2
    ]
    
    # Run multi-DataFrame optimization
    optimal_params = model.optimize_parameters_multi(
        known_values={param: params[param] for param in knowns},
        data_dfs=[piControl_data, bgc_data, full_data],
        initial_guesses=initial_guesses,
        bounds=bounds,
        co2_dfs=[None, co2_data, co2_data]  # CO2 data for bgc_data and full_data, None for piControl_data
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.6 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    return params


def run_complete_optimization(piControl_data: pd.DataFrame, full_data: pd.DataFrame, bgc_data: pd.DataFrame,
                            co2_data: pd.DataFrame, preliminary_params: Dict[str, float]) -> Dict[str, float]:
    """
    Run complete optimization using preliminary results as starting points.
    
    Args:
        piControl_data: Pre-industrial control data
        full_data: Historical + SSP585 data
        bgc_data: Historical-bgc + SSP585-bgc data
        co2_data: CO2 concentration data
        preliminary_params: Parameters from preliminary optimization
        
    Returns:
        Dictionary of final optimized parameters
    """
    print("\n=== Step 3: Complete Optimization ===")
    
    model = CoinBGC()
    
    # Set knowns and unknowns for complete optimization
    knowns = ['Ksoil_0', 'alpha']  # Only command line parameters are known
    unknowns = ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_max', 'Ktfp_co2']
    
    model.set_parameter_sets(knowns, unknowns)
    
    # Create initial guesses from preliminary results
    initial_guesses = {param: preliminary_params[param] for param in unknowns}
    
    # Create bounds (half to twice the preliminary values)
    bounds = []
    for param in unknowns:
        value = preliminary_params[param]
        bounds.append((value * 0.5, value * 2.0))
    
    # Run multi-DataFrame optimization using all datasets
    optimal_params = model.optimize_parameters_multi(
        known_values={param: preliminary_params[param] for param in knowns},
        data_dfs=[piControl_data, bgc_data, full_data],
        initial_guesses=initial_guesses,
        bounds=bounds,
        co2_dfs=[None, co2_data, co2_data]  # CO2 data for bgc_data and full_data, None for piControl_data
    )
    
    # Combine known and optimized parameters
    final_params = {**preliminary_params, **optimal_params}
    
    print(f"Complete optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    return final_params


def run_main_analysis(regions: List[str], models: List[str], Ksoil_0: float, alpha: float) -> Dict[str, Dict[str, float]]:
    """
    Main analysis function that orchestrates the complete COIN-BGC analysis.
    
    Args:
        regions: List of regions to analyze
        models: List of models to analyze
        Ksoil_0: Soil respiration parameter
        alpha: Production function exponent
        
    Returns:
        Dictionary of results for each region/model combination
    """
    print("=== COIN-BGC Main Analysis ===")
    print(f"Regions: {regions}")
    print(f"Models: {models}")
    print(f"Ksoil_0: {Ksoil_0}")
    print(f"alpha: {alpha}")
    
    results = {}
    
    for region in regions:
        for model in models:
            print(f"\n{'='*60}")
            print(f"Processing: {region} / {model}")
            print(f"{'='*60}")
            
            # Step 1: Read in data
            print("\nStep 1: Loading data...")
            piControl_data, full_data, bgc_data = load_data_for_analysis([region], [model])
            co2_data = load_co2_data()
            
            print(f"Loaded data:")
            print(f"  piControl_data: {len(piControl_data)} rows")
            print(f"  full_data: {len(full_data)} rows")
            print(f"  bgc_data: {len(bgc_data)} rows")
            print(f"  co2_data: {len(co2_data)} rows")
            
            # Step 2: Preliminary optimizations
            print("\nStep 2: Running preliminary optimizations...")
            preliminary_params = run_preliminary_optimizations(
                piControl_data, full_data, bgc_data, co2_data, Ksoil_0, alpha
            )
            
            # Step 3: Complete optimization
            print("\nStep 3: Running complete optimization...")
            final_params = run_complete_optimization(
                piControl_data, full_data, bgc_data, co2_data, preliminary_params
            )
            
            # Store results
            results[f"{region}_{model}"] = final_params
            
            print(f"\nFinal parameters for {region} / {model}:")
            for param, value in final_params.items():
                print(f"  {param}: {value:.6f}")
    
    return results


def example_usage():
    """
    Example usage of the COIN-BGC system.
    
    This demonstrates how to use the clean architecture with the three lists
    of keys and the two core routines.
    """
    # Initialize the system
    model = CoinBGC()
    
    # Define known and unknown parameters
    knowns = ['Ksoil_0', 'alpha']  # User specifies these
    unknowns = ['Ktfp_0', 'Kresp_0', 'Ktfp_co2']  # System optimizes these
    
    # Set the parameter sets
    model.set_parameter_sets(knowns, unknowns)
    
    # Create sample data (replace with actual data loading)
    sample_data = pd.DataFrame({
        'year': [1850, 1851, 1852],
        'tas': [20.0, 20.1, 20.2],
        'pr': [3.0, 3.1, 3.2],
        'npp': [100.0, 101.0, 102.0],
        'gpp': [200.0, 202.0, 204.0],
        'co2': [284.0, 284.1, 284.2]
    })
    
    # Define known parameter values
    known_values = {
        'Ksoil_0': 0.1,
        'alpha': 0.5,
        'Ktfp_0': 2.0,
        'Cland_0': 100.0,
        'Ktfp_co2': 10.0,
        'Ktfp_co2_max': 1.0
    }
    
    # Example 1: Basic model execution
    print("=== Example 1: Basic Model Execution ===")
    results = model.execute_model(sample_data, known_values)
    print("Model execution results:")
    print(results[['year', 'gpp_data', 'GPP_model', 'Cland']].head())
    
    # Example 2: Parameter optimization
    print("\n=== Example 2: Parameter Optimization ===")
    optimal_unknowns = model.optimize_parameters(known_values, sample_data)
    print("Optimal unknown parameters:")
    for param, value in optimal_unknowns.items():
        print(f"  {param}: {value:.6f}")
    
    # Example 3: Run model with optimized parameters
    print("\n=== Example 3: Model with Optimized Parameters ===")
    optimized_results = model.execute_model(sample_data, known_values)
    print("Optimized model results:")
    print(optimized_results[['year', 'gpp_data', 'GPP_model', 'Cland']].head())


if __name__ == "__main__":
    example_usage()
