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
import os
from datetime import datetime


# Global timestamp for the current run
_current_run_timestamp = None


def setup_output_directory():
    """Create output directory if it doesn't exist and return the path."""
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_run_output_directory():
    """Get the timestamped output directory for the current run."""
    global _current_run_timestamp
    
    if _current_run_timestamp is None:
        _current_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_output_dir = setup_output_directory()
    run_output_dir = os.path.join(base_output_dir, f"run_{_current_run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


def reset_run_timestamp():
    """Reset the run timestamp to create a new run directory."""
    global _current_run_timestamp
    _current_run_timestamp = None


def save_fitted_parameters(all_fitted_params, step, region=None, model=None):
    """Save fitted parameters to CSV file with standardized column order."""
    output_dir = get_run_output_directory()
    timestamp = _current_run_timestamp
    
    # Define standardized column order
    standard_columns = [
        'step', 'model', 'region', 'optimization_success',
        'final_mse',
        'Ksoil_0', 'Kresp_0', 'Ktfp_0', 'alpha', 'Cland_0', 'Ktfp_co2', 'Ktfp_co2_max',
        'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2'
    ]
    
    # Convert results to DataFrame format
    all_params_list = []
    for key, params in all_fitted_params.items():
        # If region and model are provided, use them; otherwise parse from key
        if region is not None and model is not None:
            current_region, current_model = region, model
        elif '_' in key and not key.startswith('step'):
            # Parse region and model from key (for main analysis results)
            current_region, current_model = key.rsplit('_', 1)
        else:
            # For step keys like "step2_1", use the provided region/model or defaults
            current_region = region if region is not None else 'unknown'
            current_model = model if model is not None else 'unknown'
        
        # Create parameter row
        param_row = {
            'step': step,
            'model': current_model,
            'region': current_region,
            'optimization_success': True,
            'final_mse': 0.0  # Will be updated if available
        }
        param_row.update(params)
        all_params_list.append(param_row)
    
    all_params_df = pd.DataFrame(all_params_list)
    
    # Ensure all standard columns exist
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
    
    return filepath


def save_simulation_results(results_dict, step):
    """Save simulation results to CSV files."""
    output_dir = get_run_output_directory()
    timestamp = _current_run_timestamp
    
    for key, results_df in results_dict.items():
        if '_' in key:
            region, model = key.rsplit('_', 1)
        else:
            region, model = key, 'unknown'
        
        filename = f"simulation_results_{region}_{model}_{step}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        results_df.to_csv(filepath, index=False)
        print(f"Simulation results saved to: {filepath}")


def create_pdf_books():
    """Create PDF books for visualization."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    output_dir = get_run_output_directory()
    timestamp = _current_run_timestamp
    
    print("Creating PDF books for all simulation stages...")
    
    # Find all simulation results files
    simulation_files = []
    if os.path.isdir(output_dir):
        simulation_files = [f for f in os.listdir(output_dir) if f.startswith("simulation_") and f.endswith(f"{timestamp}.csv")]
    
    if not simulation_files:
        print("No simulation results found.")
        return
    
    # Group files by stage and region/model
    stage_files = {}
    for file_path in simulation_files:
        # Extract stage and region/model from filename
        # Format: simulation_step2_1_timestamp.csv or simulation_region_model_dataset_timestamp.csv
        if file_path.startswith("simulation_step"):
            # Step simulation files
            parts = file_path.replace("simulation_", "").replace(f"_{timestamp}.csv", "").split("_")
            stage = f"step{parts[0]}_{parts[1]}"  # step2_1, step2_2, etc.
            if stage not in stage_files:
                stage_files[stage] = []
            stage_files[stage].append(file_path)
        else:
            # Final simulation files (region_model_dataset)
            parts = file_path.replace("simulation_", "").replace(f"_{timestamp}.csv", "").split("_")
            if len(parts) >= 3:
                region = parts[0]
                model = "_".join(parts[1:-1])
                dataset = parts[-1]
                stage = f"final_{dataset}"  # final_piControl, final_full, final_bgc
                if stage not in stage_files:
                    stage_files[stage] = []
                stage_files[stage].append(file_path)
    
    # Create PDF book for each stage
    for stage, files in stage_files.items():
        print(f"Creating {stage} book...")
        pdf_path = os.path.join(output_dir, f"{stage.capitalize()}_Results_{timestamp}.pdf")
        
        with PdfPages(pdf_path) as pdf:
            for file_path in sorted(files):
                # Load simulation results
                full_path = os.path.join(output_dir, file_path)
                df = pd.read_csv(full_path)
                
                # Extract region and model from data
                region = df['region'].iloc[0] if 'region' in df.columns else 'Unknown'
                model = df['model'].iloc[0] if 'model' in df.columns else 'Unknown'
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot GPP data and model
                if 'gpp_data' in df.columns and 'GPP_model' in df.columns:
                    ax.plot(df['year'], df['gpp_data'], 'k-', linewidth=2, label='GPP Data', alpha=0.8)
                    ax.plot(df['year'], df['GPP_model'], 'b-', linewidth=1, label='GPP Model', alpha=0.6)
                    
                    # Calculate MSE
                    mse = ((df['gpp_data'] - df['GPP_model'])**2).mean()
                    
                    # Customize plot
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
                    ax.set_title(f'{stage.replace("_", " ").title()}: {region} / {model}', fontsize=14, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Add MSE information
                    ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        print(f"{stage.capitalize()} book saved to: {pdf_path}")
    
    # Create BGC vs Full comparison book (like old Step 3 vs Step 4)
    print("Creating BGC vs Full comparison book...")
    bgc_full_pdf_path = os.path.join(output_dir, f"BGC_vs_Full_Comparison_{timestamp}.pdf")
    with PdfPages(bgc_full_pdf_path) as pdf:
        # Group final files by region/model
        final_files = {}
        for file_path in simulation_files:
            if not file_path.startswith("simulation_step"):
                parts = file_path.replace("simulation_", "").replace(f"_{timestamp}.csv", "").split("_")
                if len(parts) >= 3:
                    region = parts[0]
                    model = "_".join(parts[1:-1])
                    dataset = parts[-1]
                    key = f"{region}_{model}"
                    if key not in final_files:
                        final_files[key] = {}
                    final_files[key][dataset] = file_path
        
        for region_model, datasets in final_files.items():
            region, model = region_model.split("_", 1)
            
            # Check if we have both bgc and full datasets
            if 'bgc' not in datasets or 'full' not in datasets:
                print(f"Warning: Missing BGC or Full results for {region} / {model}")
                continue
            
            # Load both datasets
            bgc_path = os.path.join(output_dir, datasets['bgc'])
            full_path = os.path.join(output_dir, datasets['full'])
            
            df_bgc = pd.read_csv(bgc_path)
            df_full = pd.read_csv(full_path)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot GPP data and models (like Step 3 vs Step 4 format)
            ax.plot(df_bgc['year'], df_bgc['gpp_data'], 'b-', linewidth=2, label='GPP Data (BGC)', alpha=0.8)
            ax.plot(df_full['year'], df_full['gpp_data'], 'r-', linewidth=2, label='GPP Data (Full)', alpha=0.8)
            ax.plot(df_bgc['year'], df_bgc['GPP_model'], 'b-', linewidth=1, label='GPP Model (BGC)', alpha=0.6)
            ax.plot(df_full['year'], df_full['GPP_model'], 'r-', linewidth=1, label='GPP Model (Full)', alpha=0.6)
            
            # Customize plot
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
            ax.set_title(f'BGC vs Full Comparison: {region} / {model}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"BGC vs Full comparison book saved to: {bgc_full_pdf_path}")


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
        co2_factor = 1.0 + params['Ktfp_co2_max'] * co2 / (co2 + params['Ktfp_co2'])
        
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


def load_intermediate_parameters(region: str, model: str) -> Dict[str, Dict[str, float]]:
    """
    Load intermediate parameters for each step from saved files.
    
    Args:
        region: Region name
        model: Model name
        
    Returns:
        Dictionary of parameters for each step
    """
    output_dir = get_run_output_directory()
    timestamp = _current_run_timestamp
    
    intermediate_params = {}
    
    # Load parameters for each step
    for step in ['step2_1', 'step2_2', 'step2_3', 'step2_4', 'step2_5', 'step2_6']:
        param_file = os.path.join(output_dir, f"fitted_parameters_{step}_{timestamp}.csv")
        if os.path.exists(param_file):
            try:
                df = pd.read_csv(param_file)
                # Find the row for this region/model
                mask = (df['region'] == region) & (df['model'] == model)
                if mask.any():
                    row = df[mask].iloc[0]
                    params = {}
                    for col in df.columns:
                        if col not in ['step', 'model', 'region', 'optimization_success', 'final_mse']:
                            params[col] = row[col]
                    intermediate_params[step] = params
            except Exception as e:
                print(f"Warning: Could not load parameters for {step}: {e}")
    
    return intermediate_params

def run_preliminary_optimizations(piControl_data: pd.DataFrame, full_data: pd.DataFrame, bgc_data: pd.DataFrame,
                                co2_data: pd.DataFrame, Ksoil_0: float, alpha: float, region: str, model: str) -> Dict[str, float]:
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
    
    # Save Step 2.1 results (parameters only)
    step2_1_params = {
        'Kresp_0': Kresp_0,
        'Cland_0': Cland_0,
        'Ktfp_0': Ktfp_0,
        'Ktfp_tas0': tas_mean,
        'Ktfp_pr0': pr_mean
    }
    save_fitted_parameters({'step2_1': step2_1_params}, "step2_1", region, model)
    
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
        (min(Cland_0 * 0.5, Cland_0 * 2.0), max(Cland_0 * 0.5, Cland_0 * 2.0)),      # Cland_0
        (min(Ktfp_0 * 0.5, Ktfp_0 * 2.0), max(Ktfp_0 * 0.5, Ktfp_0 * 2.0)),        # Ktfp_0
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
    
    # Save Step 2.3 results
    step2_3_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2']}
    save_fitted_parameters({'step2_3': step2_3_params}, "step2_3", region, model)
    

    
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
    
    # Save Step 2.4 results
    step2_4_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_max', 'Ktfp_co2']}
    save_fitted_parameters({'step2_4': step2_4_params}, "step2_4", region, model)
    

    
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
        (min(params['Cland_0'] * 0.5, params['Cland_0'] * 2.0), max(params['Cland_0'] * 0.5, params['Cland_0'] * 2.0)),  # Cland_0
        (min(params['Ktfp_0'] * 0.5, params['Ktfp_0'] * 2.0), max(params['Ktfp_0'] * 0.5, params['Ktfp_0'] * 2.0)),    # Ktfp_0
        (params['Ktfp_tas0'] - abs(params['Ktfp_tas0']) * 0.5, params['Ktfp_tas0'] + abs(params['Ktfp_tas0']) * 0.5),  # Ktfp_tas0
        (-0.1, 0.1),                         # Ktfp_tas1
        (-0.01, 0.01),                       # Ktfp_tas2
        (min(params['Ktfp_pr0'] * 0.5, params['Ktfp_pr0'] * 2.0), max(params['Ktfp_pr0'] * 0.5, params['Ktfp_pr0'] * 2.0)),  # Ktfp_pr0
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
    
    # Save Step 2.5 results
    step2_5_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_max', 'Ktfp_co2']}
    save_fitted_parameters({'step2_5': step2_5_params}, "step2_5", region, model)
    

    
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
    
    # Save Step 2.6 results
    step2_6_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_max', 'Ktfp_co2']}
    save_fitted_parameters({'step2_6': step2_6_params}, "step2_6", region, model)
    

    
    return params


def run_complete_optimization(piControl_data: pd.DataFrame, full_data: pd.DataFrame, bgc_data: pd.DataFrame,
                            co2_data: pd.DataFrame, preliminary_params: Dict[str, float], region: str, model: str) -> Dict[str, float]:
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
    
    # Create bounds based on parameter physical constraints
    bounds = []
    for param in unknowns:
        value = preliminary_params[param]
        
        if param == 'Kresp_0':
            # Plant respiration fraction: 0.1 to 0.9
            bounds.append((0.1, 0.9))
        elif param in ['Cland_0', 'Ktfp_0', 'Ktfp_pr0', 'Ktfp_co2', 'Ktfp_co2_max']:
            # Must be positive: half to twice the value
            lower = min(value * 0.5, value * 2.0)
            upper = max(value * 0.5, value * 2.0)
            bounds.append((lower, upper))
        elif param == 'Ktfp_tas0':
            # Reference temperature: can be negative, use wider bounds
            lower = value - abs(value) * 0.5
            upper = value + abs(value) * 0.5
            bounds.append((lower, upper))
        elif param in ['Ktfp_tas1', 'Ktfp_pr1']:
            # Climate sensitivity: can be negative, use fixed bounds
            if param in ['Ktfp_tas1', 'Ktfp_pr1']:
                bounds.append((-0.1, 0.1))  # Linear terms
            else:
                bounds.append((-0.01, 0.01))  # Quadratic terms
        else:
            # Default: half to twice the value
            lower = min(value * 0.5, value * 2.0)
            upper = max(value * 0.5, value * 2.0)
            bounds.append((lower, upper))
    
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
    
    # Save Step 3 results
    step3_params = {param: final_params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_max', 'Ktfp_co2']}
    save_fitted_parameters({'step3': step3_params}, "step3", region, model)
    
    return final_params


def run_all_simulations(optimization_results: Dict[str, Dict[str, float]], regions: List[str], models: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run all forward simulations for all regions/models and all steps.
    
    Args:
        optimization_results: Dictionary of optimization results for each region/model
        regions: List of regions
        models: List of models
        
    Returns:
        Dictionary of simulation results organized by region_model -> step -> DataFrame
    """
    print("=== Running All Forward Simulations ===")
    
    simulation_results = {}
    
    for region in regions:
        for model in models:
            region_model_key = f"{region}_{model}"
            print(f"\nProcessing simulations for: {region} / {model}")
            
            # Load data for this region/model
            piControl_data, full_data, bgc_data = load_data_for_analysis([region], [model])
            co2_data = load_co2_data()
            
            # Get final parameters for this region/model
            if region_model_key not in optimization_results:
                print(f"Warning: No optimization results found for {region_model_key}")
                continue
                
            final_params = optimization_results[region_model_key]
            
            # Initialize simulation results for this region/model
            simulation_results[region_model_key] = {}
            
            # Create model instance
            model_instance = CoinBGC()
            
            # Step 2.1: Initial parameters from historical data (simulate with piControl data)
            print("  Running Step 2.1 simulation...")
            historical_data = full_data[full_data['year'] <= 2014].copy()
            gpp_mean = historical_data['gpp'].mean()
            npp_mean = historical_data['npp'].mean()
            tas_mean = historical_data['tas'].mean()
            pr_mean = historical_data['pr'].mean()
            
            Kresp_0 = npp_mean / gpp_mean
            Cland_0 = npp_mean / final_params['Ksoil_0']
            Ktfp_0 = gpp_mean / (Cland_0 ** final_params['alpha'])
            
            step2_1_params = {
                'Ksoil_0': final_params['Ksoil_0'],
                'alpha': final_params['alpha'],
                'Kresp_0': Kresp_0,
                'Cland_0': Cland_0,
                'Ktfp_0': Ktfp_0,
                'Ktfp_tas0': tas_mean,
                'Ktfp_pr0': pr_mean,
                'Ktfp_tas1': 0.0,
                'Ktfp_tas2': 0.0,
                'Ktfp_pr1': 0.0,
                'Ktfp_pr2': 0.0,
                'Ktfp_co2_max': 1.0,
                'Ktfp_co2': 10.0
            }
            
            step2_1_sim = model_instance.execute_model(piControl_data, step2_1_params)
            step2_1_sim['region'] = region
            step2_1_sim['model'] = model
            simulation_results[region_model_key]['step2_1'] = step2_1_sim
            
            # Step 2.2: With reference values set to historical means (simulate with piControl data)
            print("  Running Step 2.2 simulation...")
            step2_2_params = step2_1_params.copy()
            step2_2_params['Ktfp_tas0'] = tas_mean
            step2_2_params['Ktfp_pr0'] = pr_mean
            
            step2_2_sim = model_instance.execute_model(piControl_data, step2_2_params)
            step2_2_sim['region'] = region
            step2_2_sim['model'] = model
            simulation_results[region_model_key]['step2_2'] = step2_2_sim
            
            # Load intermediate parameters for accurate step simulations
            intermediate_params = load_intermediate_parameters(region, model)
            
            # Step 2.3: With optimized climate sensitivity parameters (simulate with piControl data)
            print("  Running Step 2.3 simulation...")
            if 'step2_3' in intermediate_params:
                step2_3_params = {**final_params, **intermediate_params['step2_3']}
            else:
                step2_3_params = final_params.copy()
            step2_3_sim = model_instance.execute_model(piControl_data, step2_3_params)
            step2_3_sim['region'] = region
            step2_3_sim['model'] = model
            simulation_results[region_model_key]['step2_3'] = step2_3_sim
            
            # Step 2.4: With optimized CO2 parameters
            print("  Running Step 2.4 simulation...")
            if 'step2_4' in intermediate_params:
                step2_4_params = {**final_params, **intermediate_params['step2_4']}
            else:
                step2_4_params = final_params.copy()
            step2_4_sim = model_instance.execute_model(full_data, step2_4_params, co2_data)
            step2_4_sim['region'] = region
            step2_4_sim['model'] = model
            simulation_results[region_model_key]['step2_4'] = step2_4_sim
            
            # Step 2.5: With all parameters optimized
            print("  Running Step 2.5 simulation...")
            if 'step2_5' in intermediate_params:
                step2_5_params = {**final_params, **intermediate_params['step2_5']}
            else:
                step2_5_params = final_params.copy()
            step2_5_sim = model_instance.execute_model(full_data, step2_5_params, co2_data)
            step2_5_sim['region'] = region
            step2_5_sim['model'] = model
            simulation_results[region_model_key]['step2_5'] = step2_5_sim
            
            # Step 2.6: Final optimization
            print("  Running Step 2.6 simulation...")
            if 'step2_6' in intermediate_params:
                step2_6_params = {**final_params, **intermediate_params['step2_6']}
            else:
                step2_6_params = final_params.copy()
            step2_6_sim = model_instance.execute_model(full_data, step2_6_params, co2_data)
            step2_6_sim['region'] = region
            step2_6_sim['model'] = model
            simulation_results[region_model_key]['step2_6'] = step2_6_sim
            
            # Final simulations: piControl, full, bgc
            print("  Running final simulations...")
            piControl_sim = model_instance.execute_model(piControl_data, final_params)
            piControl_sim['region'] = region
            piControl_sim['model'] = model
            simulation_results[region_model_key]['piControl'] = piControl_sim
            
            full_sim = model_instance.execute_model(full_data, final_params, co2_data)
            full_sim['region'] = region
            full_sim['model'] = model
            simulation_results[region_model_key]['full'] = full_sim
            
            bgc_sim = model_instance.execute_model(bgc_data, final_params, co2_data)
            bgc_sim['region'] = region
            bgc_sim['model'] = model
            simulation_results[region_model_key]['bgc'] = bgc_sim
    
    return simulation_results


def generate_all_outputs(simulation_results: Dict[str, Dict[str, pd.DataFrame]]):
    """
    Generate all CSV files and PDF books from simulation results.
    
    Args:
        simulation_results: Dictionary of simulation results from run_all_simulations
    """
    print("=== Generating All Outputs ===")
    
    output_dir = get_run_output_directory()
    timestamp = _current_run_timestamp
    
    # Generate CSV files
    print("Generating CSV files...")
    for region_model_key, step_results in simulation_results.items():
        region, model = region_model_key.split("_", 1)
        
        for step_name, sim_df in step_results.items():
            # Create filename with region/model information
            if step_name.startswith('step'):
                filename = f"simulation_{region}_{model}_{step_name}_{timestamp}.csv"
            else:
                filename = f"simulation_{region}_{model}_{step_name}_{timestamp}.csv"
            
            filepath = os.path.join(output_dir, filename)
            sim_df.to_csv(filepath, index=False)
            print(f"  Saved: {filename}")
    
    # Generate PDF books
    print("Generating PDF books...")
    create_pdf_books_from_simulation_results(simulation_results)


def create_pdf_books_from_simulation_results(simulation_results: Dict[str, Dict[str, pd.DataFrame]]):
    """
    Create PDF books from simulation results.
    
    Args:
        simulation_results: Dictionary of simulation results from run_all_simulations
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    output_dir = get_run_output_directory()
    timestamp = _current_run_timestamp
    
    # Group simulations by step
    step_simulations = {}
    for region_model_key, step_results in simulation_results.items():
        for step_name, sim_df in step_results.items():
            if step_name not in step_simulations:
                step_simulations[step_name] = []
            step_simulations[step_name].append((region_model_key, sim_df))
    
    # Create PDF book for each step
    for step_name, simulations in step_simulations.items():
        print(f"Creating {step_name} book...")
        pdf_path = os.path.join(output_dir, f"{step_name.capitalize()}_Results_{timestamp}.pdf")
        
        with PdfPages(pdf_path) as pdf:
            for region_model_key, sim_df in sorted(simulations):
                region = sim_df['region'].iloc[0]
                model = sim_df['model'].iloc[0]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot GPP data and model
                if 'gpp_data' in sim_df.columns and 'GPP_model' in sim_df.columns:
                    ax.plot(sim_df['year'], sim_df['gpp_data'], 'k-', linewidth=2, label='GPP Data', alpha=0.8)
                    ax.plot(sim_df['year'], sim_df['GPP_model'], 'b-', linewidth=1, label='GPP Model', alpha=0.6)
                    
                    # Calculate MSE
                    mse = ((sim_df['gpp_data'] - sim_df['GPP_model'])**2).mean()
                    
                    # Customize plot
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
                    ax.set_title(f'{step_name.replace("_", " ").title()}: {region} / {model}', fontsize=14, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Add MSE information
                    ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        print(f"{step_name.capitalize()} book saved to: {pdf_path}")
    
    # Create BGC vs Full comparison book
    print("Creating BGC vs Full comparison book...")
    bgc_full_pdf_path = os.path.join(output_dir, f"BGC_vs_Full_Comparison_{timestamp}.pdf")
    with PdfPages(bgc_full_pdf_path) as pdf:
        for region_model_key, step_results in simulation_results.items():
            if 'bgc' in step_results and 'full' in step_results:
                region = step_results['bgc']['region'].iloc[0]
                model = step_results['bgc']['model'].iloc[0]
                
                df_bgc = step_results['bgc']
                df_full = step_results['full']
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot GPP data and models
                ax.plot(df_bgc['year'], df_bgc['gpp_data'], 'b-', linewidth=2, label='GPP Data (BGC)', alpha=0.8)
                ax.plot(df_full['year'], df_full['gpp_data'], 'r-', linewidth=2, label='GPP Data (Full)', alpha=0.8)
                ax.plot(df_bgc['year'], df_bgc['GPP_model'], 'b-', linewidth=1, label='GPP Model (BGC)', alpha=0.6)
                ax.plot(df_full['year'], df_full['GPP_model'], 'r-', linewidth=1, label='GPP Model (Full)', alpha=0.6)
                
                # Customize plot
                ax.set_xlabel('Year', fontsize=12)
                ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
                ax.set_title(f'BGC vs Full Comparison: {region} / {model}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
    
    print(f"BGC vs Full comparison book saved to: {bgc_full_pdf_path}")


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
    
    # Setup output environment
    reset_run_timestamp()
    output_dir = get_run_output_directory()
    print(f"Output directory: {output_dir}")
    
    results = {}
    
    # Phase 1: Run all optimizations (save only parameters)
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
            
            # Step 2: Preliminary optimizations (save parameters only)
            print("\nStep 2: Running preliminary optimizations...")
            preliminary_params = run_preliminary_optimizations(
                piControl_data, full_data, bgc_data, co2_data, Ksoil_0, alpha, region, model
            )
            
            # Step 3: Complete optimization
            print("\nStep 3: Running complete optimization...")
            final_params = run_complete_optimization(
                piControl_data, full_data, bgc_data, co2_data, preliminary_params, region, model
            )
            
            # Store results
            results[f"{region}_{model}"] = final_params
            
            print(f"\nFinal parameters for {region} / {model}:")
            for param, value in final_params.items():
                print(f"  {param}: {value:.6f}")
    
    # Save final optimization results
    print(f"\n=== Saving Optimization Results ===")
    save_fitted_parameters(results, "complete")
    
    # Phase 2: Run all forward simulations
    print(f"\n=== Phase 2: Running All Forward Simulations ===")
    simulation_results = run_all_simulations(results, regions, models)
    
    # Phase 3: Generate all outputs (CSV files and PDF books)
    print(f"\n=== Phase 3: Generating All Outputs ===")
    generate_all_outputs(simulation_results)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")
    
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
