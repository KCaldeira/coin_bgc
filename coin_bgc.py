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
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
import glob
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from scipy.stats import pearsonr
import json


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
        'Ksoil_0', 'Kresp_0', 'Ktfp_0', 'alpha', 'Cland_0', 'Ktfp_co2_half',
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
    
    # For intermediate steps, include region/model in filename to avoid overwriting
    if step.startswith('step') and region is not None and model is not None:
        filename = f"fitted_parameters_{region}_{model}_{step}_{timestamp}.csv"
    else:
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
                    
                    # Set y-axis bounds ensuring 0 is on the chart
                    all_data_values = list(df['gpp_data']) + list(df['GPP_model'])
                    set_y_axis_bounds(ax, all_data_values)
                    
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
            
            # Set y-axis bounds ensuring 0 is on the chart
            all_data_values = (list(df_bgc['gpp_data']) + list(df_bgc['GPP_model']) + 
                             list(df_full['gpp_data']) + list(df_full['GPP_model']))
            set_y_axis_bounds(ax, all_data_values)
            
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
            'Ktfp_co2_half',     # CO2 fertilization sensitivity
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
        # Calculate Ktfp_co2_max such that co2_factor = 1 when co2 = co2_0
        # Ktfp_co2_max = (co2_0 + Ktfp_co2_half) / co2_0
        Ktfp_co2_max = (self.co2_0 + params['Ktfp_co2_half']) / self.co2_0
        co2_factor = 1.0 + Ktfp_co2_max * co2 / (co2 + params['Ktfp_co2_half'])
        
        return np.maximum(0.0, ktfp_0 * tas_factor * pr_factor * co2_factor)
    
    def execute_model(self, data_df: pd.DataFrame, params: Dict[str, float], 
                     co2_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Basic Model Execution Routine: Execute the model from start year to end year.
        
        This is the first core routine that takes a pandas DataFrame and a complete
        parameter dictionary.
        
        Args:
            data_df: DataFrame with columns ['year', 'tas', 'pr', 'npp', 'gpp']
            params: Complete parameter dictionary with all parameters
            co2_df: Optional DataFrame with CO2 concentration data
            
        Returns:
            DataFrame with model results including calculated GPP, NPP, Cland, etc.
        """
        # Use the provided complete parameter dictionary
        
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
            
            # Store results for this year with complete parameter universe
            result_row = {
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
                # Include complete parameter universe for transparency
                'Ksoil_0': params['Ksoil_0'],
                'Kresp_0': params['Kresp_0'],
                'Ktfp_0': params['Ktfp_0'],
                'alpha': params['alpha'],
                'Cland_0': params['Cland_0'],
                'Ktfp_co2_half': params['Ktfp_co2_half'],
                'Ktfp_tas0': params['Ktfp_tas0'],
                'Ktfp_tas1': params['Ktfp_tas1'],
                'Ktfp_tas2': params['Ktfp_tas2'],
                'Ktfp_pr0': params['Ktfp_pr0'],
                'Ktfp_pr1': params['Ktfp_pr1'],
                'Ktfp_pr2': params['Ktfp_pr2'],
                # Also include the calculated values for verification
                'Ksoil': Ksoil,
                'Kresp': Kresp,
                'Ktfp': Ktfp
            }
            results.append(result_row)
            
            # Update Cland for next year
            Cland = Cland + dCland_dt
            
            # Check if Cland has gone negative or zero - if so, stop simulation
            if Cland <= 0:
                print(f"Warning: Cland went to {Cland:.6f} at year {year}. Stopping simulation.")
                print(f"Parameters that caused collapse:")
                print(f"  Ksoil_0: {params['Ksoil_0']:.6f}")
                print(f"  alpha: {params['alpha']:.6f}")
                print(f"  Kresp_0: {params['Kresp_0']:.6f}")
                print(f"  Cland_0: {params['Cland_0']:.6f}")
                print(f"  Ktfp_0: {params['Ktfp_0']:.6f}")
                print(f"  Ktfp_tas0: {params['Ktfp_tas0']:.6f}")
                print(f"  Ktfp_tas1: {params['Ktfp_tas1']:.6f}")
                print(f"  Ktfp_tas2: {params['Ktfp_tas2']:.6f}")
                print(f"  Ktfp_pr0: {params['Ktfp_pr0']:.6f}")
                print(f"  Ktfp_pr1: {params['Ktfp_pr1']:.6f}")
                print(f"  Ktfp_pr2: {params['Ktfp_pr2']:.6f}")
                print(f"  Ktfp_co2_half: {params['Ktfp_co2_half']:.6f}")
                print(f"Current year values:")
                print(f"  tas: {tas:.6f}")
                print(f"  pr: {pr:.6f}")
                print(f"  co2: {co2:.6f}")
                print(f"  Ktfp: {Ktfp:.6f}")
                print(f"  GPP: {GPP:.6f}")
                print(f"  NPP: {NPP:.6f}")
                print(f"  Sresp: {Sresp:.6f}")
                print(f"  dCland_dt: {dCland_dt:.6f}")
                print(f"  Previous Cland: {Cland - dCland_dt:.6f}")
                # Fill remaining rows with zeros
                for j in range(i + 1, len(data_df)):
                    remaining_row = data_df.iloc[j]
                    zero_row = {
                        'year': remaining_row['year'],
                        'Cland': 0.0,
                        'GPP_model': 0.0,
                        'NPP_model': 0.0,
                        'SOILresp': 0.0,
                        'dCland_dt': 0.0,
                        'tas_data': remaining_row['tas'],
                        'pr_data': remaining_row['pr'],
                        'co2': self._get_co2_for_year(co2_df, remaining_row['year']) if co2_df is not None else self.co2_0,
                        'gpp_data': remaining_row['gpp'],
                        'npp_data': remaining_row['npp'],
                        'region': remaining_row.get('region', 'unknown'),
                        'model': remaining_row.get('model', 'unknown'),
                        # Include complete parameter universe for transparency
                        'Ksoil_0': params['Ksoil_0'],
                        'Kresp_0': params['Kresp_0'],
                        'Ktfp_0': params['Ktfp_0'],
                        'alpha': params['alpha'],
                        'Cland_0': params['Cland_0'],
                        'Ktfp_co2_half': params['Ktfp_co2_half'],
                        'Ktfp_tas0': params['Ktfp_tas0'],
                        'Ktfp_tas1': params['Ktfp_tas1'],
                        'Ktfp_tas2': params['Ktfp_tas2'],
                        'Ktfp_pr0': params['Ktfp_pr0'],
                        'Ktfp_pr1': params['Ktfp_pr1'],
                        'Ktfp_pr2': params['Ktfp_pr2'],
                        # Also include the calculated values for verification
                        'Ksoil': Ksoil,
                        'Kresp': Kresp,
                        'Ktfp': 0.0
                    }
                    results.append(zero_row)
                break
        
        return pd.DataFrame(results)
    
    

    
    def objective_function_multi(self, unknown_values: np.ndarray, knowns: Dict[str, float], 
                                data_dfs: List[pd.DataFrame], co2_dfs: Optional[List[pd.DataFrame]] = None) -> float:
        """
        Objective function for multi-DataFrame optimization: Combined MSE across all DataFrames.
        
        Args:
            unknown_values: Array of unknown parameter values
            knowns: Dictionary of known parameter values
            data_dfs: List of DataFrames with observed data
            co2_dfs: Optional list of DataFrames with CO2 concentration data
            
        Returns:
            Combined mean squared error across all DataFrames
        """
        # Track function evaluations for max_iterations enforcement
        if not hasattr(self, '_function_eval_count'):
            self._function_eval_count = 0
        self._function_eval_count += 1
        
        # Check if we've exceeded max function evaluations
        if hasattr(self, '_max_function_evals') and self._function_eval_count > self._max_function_evals:
            print(f"  WARNING: Exceeded {self._max_function_evals} function evaluations. Returning high penalty.")
            return 1e6  # Return very high value to force termination
        
        # Convert unknown_values array to dictionary
        unknown_dict = dict(zip(self.unknowns, unknown_values))
        
        # Create complete parameter dictionary
        params = self.create_parameter_dict(knowns, unknown_dict)
        
        total_mse = 0.0
        total_points = 0
        
        # Run model for each DataFrame and accumulate MSE
        for i, data_df in enumerate(data_dfs):
            # Get corresponding CO2 DataFrame
            co2_df = co2_dfs[i]
            
            # Run model for this DataFrame
            results = self.execute_model(data_df, params, co2_df)
            
            # Calculate MSE for this DataFrame
            mse = np.mean((results['gpp_data'] - results['GPP_model']) ** 2)
            n_points = len(results)
            
            # Accumulate weighted MSE
            total_mse += mse * n_points
            total_points += n_points
        
        # Return average MSE across all DataFrames
        return total_mse / total_points
    
    def optimize_parameters(self, knowns: Dict[str, float], unknowns: Dict[str, List[float]], 
                                data_dfs: List[pd.DataFrame], co2_dfs: Optional[List[pd.DataFrame]] = None,
                                max_iterations: int = 100000) -> Dict[str, float]:
        """
        Optimization Routine: Optimize for the unknowns given the knowns.
        
        This optimization routine optimizes for the unknowns given the knowns and one or more
        pandas DataFrames. For multiple DataFrames, the model is run individually for each 
        DataFrame using the same set of parameters, and the objective function minimizes 
        the combined MSE across all simulations.
        
        Args:
            knowns: Dictionary of known parameter values {param_name: value}
            unknowns: Dictionary of unknown parameters with [lower_bound, initial_guess, upper_bound]
            data_dfs: List of DataFrames with observed data (can be single DataFrame in a list)
            co2_dfs: Optional list of DataFrames with CO2 concentration data
            max_iterations: Maximum number of iterations for optimization (default: 100000)
            
        Returns:
            Dictionary of optimal values for unknown parameters
        """
        print(f"  🔧 Optimize_parameters called with {len(data_dfs)} datasets, {len(unknowns)} unknowns")
        
        # Set parameter sets for this optimization
        self.set_parameter_sets(list(knowns.keys()), list(unknowns.keys()))
        print(f"  🔧 Parameter sets configured: {len(self.knowns)} knowns, {len(self.unknowns)} unknowns")
        
        # Reset function evaluation counter and set max
        self._function_eval_count = 0
        self._max_function_evals = max_iterations * 10  # Allow 10x more function evaluations than iterations
        
        # Create initial guess array
        x0 = np.array([unknowns[param][1] for param in self.unknowns])
        
        # Create bounds array
        bounds_array = [(unknowns[param][0], unknowns[param][2]) for param in self.unknowns]
        
        # Run optimization
        result = minimize(
            fun=self.objective_function_multi,
            x0=x0,
            args=(knowns, data_dfs, co2_dfs),
            bounds=bounds_array,
            method='L-BFGS-B',
            options={
                'maxiter': max_iterations,
                'gtol': 1e-8,      # Gradient tolerance (default: 1e-5)
                'ftol': 1e-10,     # Function tolerance (default: 1e-5)
                'eps': 1e-8        # Step size for finite difference (default: 1e-8)
            }
        )
        
        print(f"Multi-optimization result:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  Iterations: {result.nit}")
        print(f"  Final objective value: {result.fun}")
        print(f"  Final gradient norm: {np.linalg.norm(result.jac) if hasattr(result, 'jac') and result.jac is not None else 'N/A'}")
        print(f"  Parameter changes from initial:")
        for i, param in enumerate(self.unknowns):
            initial_val = x0[i]
            final_val = result.x[i]
            change = final_val - initial_val
            print(f"    {param}: {initial_val:.6f} -> {final_val:.6f} (change: {change:+.6f})")
        
        if not result.success:
            print(f"  WARNING: Multi-optimization did not converge to desired accuracy!")
            print(f"  Best solution found will be used, but may not be optimal.")
        
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
        
        return co2_df.loc[closest_idx, 'pco2 (ppm)']

    def execute_workflow_step(self, step, region: str, model: str, datasets: Dict, previous_results: Dict, global_params: Dict = None, bounds: Dict = None, verbose: bool = False) -> Dict:
        """
        Execute a single workflow step based on JSON configuration.
        
        This is the key method that makes the workflow flexible and JSON-driven.
        It handles different step types: calculation, optimization, and multi_optimization.
        
        Args:
            step: WorkflowStep object from configuration
            region: Region name to process
            model: Model name to process  
            datasets: Dictionary of loaded datasets
            previous_results: Results from previous workflow steps
            
        Returns:
            Dictionary containing step results (optimized parameters, calculations, etc.)
        """
        print(f"        🔧 Executing {step.step_type} step: {step.name}")
        
        # Import here to avoid circular imports
        from workflow_config import WorkflowExecutor
        
        # Get region/model specific data
        step_datasets = self._filter_datasets_for_region_model(datasets, step.data_sources, region, model)
        
        if step.step_type == "calculation":
            return self._execute_calculation_step(step, step_datasets, previous_results, global_params)
        elif step.step_type == "optimization":
            return self._execute_optimization_step(step, step_datasets, previous_results, region, model, global_params, datasets, bounds, verbose)
        else:
            raise ValueError(f"Unknown step type: {step.step_type}")
    
    def _filter_datasets_for_region_model(self, datasets: Dict, data_sources: List[str], region: str, model: str) -> Dict:
        """Filter datasets to only include data for the specified region and model."""
        filtered = {}
        
        for source in data_sources:
            if source in datasets:
                df = datasets[source]
                # Filter by region and model if columns exist
                if 'region' in df.columns and 'model' in df.columns:
                    filtered_df = df[(df['region'] == region) & (df['model'] == model)]
                    if not filtered_df.empty:
                        filtered[source] = filtered_df
                    else:
                        print(f"          ⚠️  No data found for {region}/{model} in {source}")
                else:
                    # For datasets without region/model columns (like CO2 data)
                    filtered[source] = df
        
        return filtered
    
    def _execute_calculation_step(self, step, datasets: Dict, previous_results: Dict, global_params: Dict) -> Dict:
        """Execute a calculation step (like step2_1 in the original workflow)."""
        results = {}
        
        # For calculation steps, we typically calculate parameters from data
        if step.name == "step2_1":
            # This is the original step2_1 calculation logic
            if "piControl" in datasets:
                piControl_data = datasets["piControl"]
                if not piControl_data.empty:
                    # Calculate Kresp_0 and Cland_0 as in the original implementation
                    npp_mean = piControl_data['npp'].mean()
                    gpp_mean = piControl_data['gpp'].mean() 
                    
                    # Get Ksoil_0 from global parameters - must be provided
                    if not global_params or 'Ksoil_0' not in global_params:
                        raise ValueError("Ksoil_0 must be provided in global parameters for calculation steps")
                    Ksoil_0 = global_params['Ksoil_0']
                    
                    Kresp_0 = 1 - (npp_mean / gpp_mean)
                    Cland_0 = npp_mean / Ksoil_0
                    
                    # Calculate reference temperature and precipitation (historical means)
                    tas_mean = piControl_data['tas'].mean()
                    pr_mean = piControl_data['pr'].mean()
                    
                    # Calculate Ktfp_0 using the formula from JSON: gpp_mean / (Cland_0 ** alpha)
                    if not global_params or 'alpha' not in global_params:
                        raise ValueError("alpha must be provided in global parameters for calculation steps")
                    alpha = global_params['alpha']
                    Ktfp_0 = gpp_mean / (Cland_0 ** alpha)
                    
                    results = {
                        'Kresp_0': Kresp_0,
                        'Cland_0': Cland_0,
                        'Ktfp_0': Ktfp_0,
                        'Ktfp_tas0': tas_mean,
                        'Ktfp_pr0': pr_mean,
                        'npp_mean': npp_mean,
                        'gpp_mean': gpp_mean
                    }
                    
                    print(f"          ✅ Calculated Kresp_0={Kresp_0:.6f}, Cland_0={Cland_0:.6f}")
        
        # Handle other calculation steps as defined in the JSON workflow
        elif hasattr(step, 'calculations'):
            raise ValueError(f"Generic calculation step evaluation not implemented for step {step.name}")
        else:
            raise ValueError(f"Unknown calculation step: {step.name}")
        
        return results
    
    def _execute_optimization_step(self, step, datasets: Dict, previous_results: Dict, region: str, model: str, global_params: Dict, original_datasets: Dict, bounds: Dict = None, verbose: bool = False) -> Dict:
        """Execute a single-dataset optimization step."""
        from workflow_config import WorkflowExecutor
        
        # Build params dictionary: known values and initial guesses for unknowns
        params = {}
        knowns = {}
        unknowns = {}
        
        # Add known parameters (fixed values)
        for param_name, param_spec in step.knowns.items():
            value = self._resolve_parameter_value(param_spec, previous_results, global_params)
            knowns[param_name] = value
            params[param_name] = value
        
        # Add unknown parameters (to be optimized)
        for param_name, param_spec in step.unknowns.items():
            # Get starting value from parameter spec
            if param_spec.source == "value":
                initial_value = param_spec.value
            else:
                initial_value = self._resolve_parameter_value(param_spec, previous_results, global_params)
                
            # Get bounds from centralized bounds dictionary
            if bounds and param_name in bounds:
                bounds_spec = bounds[param_name]
                if len(bounds_spec) != 2:
                    raise ValueError(f"Parameter {param_name} bounds must have exactly 2 values [lower, upper], got: {bounds_spec}")
                # Convert to old format: [lower, initial, upper]
                range_spec = [bounds_spec[0], initial_value, bounds_spec[1]]
                unknowns[param_name] = range_spec
                params[param_name] = initial_value
            else:
                raise ValueError(f"Parameter {param_name} in step {step.name} not found in bounds dictionary or bounds not provided")
        
        # Get all datasets for optimization (handles both single and multiple datasets)
        optimization_datasets = []
        
        for source in step.data_sources:
            if source in datasets:
                df = datasets[source]  # datasets are already filtered by region/model
                if not df.empty:
                    optimization_datasets.append(df)
        
        if not optimization_datasets:
            print(f"          ⚠️  No datasets available for optimization")
            return {}
        
        # Prepare CO2 data by interpolating global CO2 data for each dataset's years
        # The CO2 file now includes (0, 284.317) and (1849.9999, 284.317) for piControl years
        global_co2_data = original_datasets.get('co2_data', None)
        co2_dfs = []
        
        if global_co2_data is not None:
            for df in optimization_datasets:
                if 'year' in df.columns:
                    # Create interpolated CO2 data for this dataset's years
                    years = df['year'].unique()
                    co2_values = []
                    for year in years:
                        co2_val = self._get_co2_for_year(global_co2_data, year)
                        co2_values.append(co2_val)
                    
                    # Create CO2 DataFrame for this dataset
                    co2_df_for_dataset = pd.DataFrame({
                        'year': years,
                        'pco2 (ppm)': co2_values
                    })
                    co2_dfs.append(co2_df_for_dataset)
                else:
                    print(f"          ⚠️  Dataset missing 'year' column, using global CO2 data")
                    co2_dfs.append(global_co2_data)
        else:
            co2_dfs = None
        
        # Debug CO2 data preparation (only in verbose mode)
        if verbose:
            print(f"          🔍 Global CO2 data: {global_co2_data.shape if global_co2_data is not None else 'None'}")
            print(f"          🔍 CO2 dfs prepared: {len(co2_dfs) if co2_dfs else 'None'} datasets")
        
        # Run optimization using unified method
        print(f"          🚀 Starting optimization with {len(optimization_datasets)} datasets, {len(unknowns)} parameters to optimize...")
        try:
            optimized_params = self.optimize_parameters(knowns, unknowns, optimization_datasets, co2_dfs)
            dataset_desc = "dataset" if len(optimization_datasets) == 1 else f"{len(optimization_datasets)} datasets"
            print(f"          ✅ Optimization completed with {dataset_desc}, optimized {len(unknowns)} parameters")
            
            # Return complete parameter set (both known and optimized parameters)
            complete_result = {}
            complete_result.update(knowns)  # Add all known parameters
            complete_result.update(optimized_params)  # Add optimized parameters
            return complete_result
        except Exception as e:
            print(f"          ❌ Optimization failed: {str(e)}")
            return {}
    
    
    def _resolve_parameter_value(self, param_spec, previous_results: Dict, global_params: Dict):
        """Resolve a parameter value based on its specification in the workflow."""
        if hasattr(param_spec, 'source'):
            if param_spec.source == 'global':
                param_name = getattr(param_spec, 'name', None)
                if not param_name:
                    raise ValueError("Global parameter specification missing required 'name' field")
                if not global_params or param_name not in global_params:
                    raise ValueError(f"Global parameter {param_name} not found in provided global parameters")
                return global_params[param_name]
            elif param_spec.source == 'step':
                # Get value from previous step results
                step_name = param_spec.step
                if step_name not in previous_results:
                    raise ValueError(f"Step {step_name} results not found for parameter {getattr(param_spec, 'name', 'unknown')}")
                step_result = previous_results[step_name]
                param_name = getattr(param_spec, 'name', None)
                if not param_name:
                    raise ValueError(f"Parameter name not specified in step reference")
                if param_name not in step_result:
                    raise ValueError(f"Parameter {param_name} not found in step {step_name} results")
                return step_result[param_name]
            elif param_spec.source == 'value':
                # Direct value specification
                if not hasattr(param_spec, 'value'):
                    raise ValueError(f"Parameter spec with source 'value' missing required 'value' field")
                return param_spec.value
            else:
                raise ValueError(f"Unknown parameter source: {param_spec.source}")
        
        raise ValueError(f"Parameter specification missing required 'source' field")

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"    ✅ Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            print(f"    ❌ Failed to load {filepath}: {str(e)}")
            return pd.DataFrame()


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




def run_optimizations(piControl_data: pd.DataFrame, full_data: pd.DataFrame, bgc_data: pd.DataFrame,
                     co2_data: pd.DataFrame, Ksoil_0: float, alpha: float, region: str, model: str) -> Tuple[Dict[str, float], Dict[Tuple[str, str, str], Dict[str, float]]]:
    """
    Run all optimizations for a region/model combination.
    
    Args:
        piControl_data: Pre-industrial control data for this region/model
        full_data: Historical + SSP585 data for this region/model
        bgc_data: Historical-bgc + SSP585-bgc data for this region/model
        co2_data: CO2 concentration data
        Ksoil_0: Soil respiration parameter (from command line)
        alpha: Production function exponent (from command line)
        region: Region name
        model: Model name
        
    Returns:
        Tuple of (final_parameters, all_parameter_results)
        final_parameters: Dictionary of final optimized parameters
        all_parameter_results: Dictionary with (region, model, step) keys containing complete parameter sets
    """
    model_instance = CoinBGC()
    
    # Initialize the parameter dictionary that will be built incrementally
    all_parameter_results = {}
    
    # Step 2.1: Calculate Cland_0, Kresp_0, and Ktfp_0 based on mean values from piControl data
    print("=== Step 2.1: Calculating initial parameters from piControl data ===")
    
    # Use region/model-specific piControl data for initial calculations
    calc_data = piControl_data
    
    # Calculate mean values
    gpp_mean = calc_data['gpp'].mean()
    npp_mean = calc_data['npp'].mean()
    tas_mean = calc_data['tas'].mean()
    pr_mean = calc_data['pr'].mean()
    
    # Calculate initial parameters
    Kresp_0 = 1 -npp_mean / gpp_mean  # From steady-state: NPP = (1 - Kresp_0) * GPP
    Cland_0 = npp_mean / Ksoil_0   # From steady-state: NPP = Ksoil_0 * Cland_0
    Ktfp_0 = gpp_mean / (Cland_0 ** alpha)  # From production function: GPP = Ktfp_0 * Cland_0^alpha
    
    print(f"Initial parameters from piControl data:")
    print(f"  gpp_mean: {gpp_mean:.6f}")
    print(f"  npp_mean: {npp_mean:.6f}")
    print(f"  alpha: {alpha:.6f}")
    print(f"  Ksoil_0: {Ksoil_0:.6f}")
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
    
    # Add to parameter dictionary
    all_parameter_results[(region, model, 'step2_1')] = {
        'Ksoil_0': Ksoil_0,
        'alpha': alpha,
        'Kresp_0': Kresp_0,
        'Cland_0': Cland_0,
        'Ktfp_0': Ktfp_0,
        'Ktfp_tas0': tas_mean,
        'Ktfp_pr0': pr_mean,
        'Ktfp_tas1': 0.0,
        'Ktfp_tas2': 0.0,
        'Ktfp_pr1': 0.0,
        'Ktfp_pr2': 0.0,
        'Ktfp_co2_max': 0.0,
'Ktfp_co2_half': 0.0
    }
    
    # Initialize parameters dictionary
    params = {
        'Ksoil_0': Ksoil_0,
        'alpha': alpha,
        'Kresp_0': Kresp_0,
        'Cland_0': Cland_0,
        'Ktfp_0': Ktfp_0,
        'Ktfp_tas0': tas_mean,
        'Ktfp_pr0': pr_mean,
        'Ktfp_tas1': 0.0,
        'Ktfp_tas2': 0.0,
        'Ktfp_pr1': 0.0,
        'Ktfp_pr2': 0.0,
        'Ktfp_co2_half': 0.0
    }
    

    
    # Step 2.2: Optimize Cland_0, Ktfp_0, Ktfp_tas1, Ktfp_tas2, Ktfp_pr1, Ktfp_pr2 using historical data
    print("\n=== Step 2.2: Optimizing climate sensitivity parameters using historical data ===")
    
    # Use historical portion of full_data for step 2.3 optimization
    historical_data = full_data[full_data['year'] <= 2014].copy()  # Historical period
    
    # Define known and unknown parameters for this step
    knowns_dict = {
        'Ksoil_0': Ksoil_0,
        'alpha': alpha,
        'Kresp_0': Kresp_0,
        'Cland_0': Cland_0,
        'Ktfp_tas0': tas_mean,
        'Ktfp_pr0': pr_mean,
        'Ktfp_co2_half': 0.0
    }
    
    unknowns_dict = {
        'Ktfp_0': [Ktfp_0 * 0.5, Ktfp_0, Ktfp_0 * 2.0],
        'Ktfp_tas1': [-0.4, 0.001, 0.4],
        'Ktfp_tas2': [-0.4, 0.001, 0.4],
        'Ktfp_pr1': [-0.4, 0.001, 0.4],
        'Ktfp_pr2': [-0.4, 0.001, 0.4]
    }
    
    print(f"Step 2.2 optimization setup:")
    print(f"  Known parameters: {list(knowns_dict.keys())}")
    print(f"  Unknown parameters: {list(unknowns_dict.keys())}")
    print(f"  Known values: {knowns_dict}")
    print(f"  Unknown specifications: {unknowns_dict}")
    print(f"  Historical data points: {len(historical_data)}")
    
    # Run optimization
    optimal_params = model_instance.optimize_parameters(
        knowns=knowns_dict,
        unknowns=unknowns_dict,
        data_df=historical_data,
        co2_df=co2_data
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.2 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Collect Step 2.2 results
    step2_2_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2']}
    
    # Add to parameter dictionary
    all_parameter_results[(region, model, 'step2_2')] = params.copy()
    

    
    # Step 2.3: Optimize CO2 fertilization parameter using bgc_data
    print("\n=== Step 2.3: Optimizing CO2 parameters using bgc_data ===")
    
    # Define known and unknown parameters for this step
    knowns_dict = {
        'Ksoil_0': Ksoil_0,
        'alpha': alpha,
        'Kresp_0': params['Kresp_0'],
        'Cland_0': params['Cland_0'],
        'Ktfp_0': params['Ktfp_0'],
        'Ktfp_tas0': params['Ktfp_tas0'],
        'Ktfp_tas1': params['Ktfp_tas1'],
        'Ktfp_tas2': params['Ktfp_tas2'],
        'Ktfp_pr0': params['Ktfp_pr0'],
        'Ktfp_pr1': params['Ktfp_pr1'],
        'Ktfp_pr2': params['Ktfp_pr2']
    }
    
    unknowns_dict = {
        'Ktfp_co2_half': [10.0, 1000.0, 10000.0]
    }
    
    # Run optimization
    optimal_params = model_instance.optimize_parameters(
        knowns=knowns_dict,
        unknowns=unknowns_dict,
        data_df=bgc_data,
        co2_df=co2_data
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.3 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Collect Step 2.3 results
    step2_3_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_half']}
    
    # Add to parameter dictionary
    all_parameter_results[(region, model, 'step2_3')] = params.copy()
    

    
    # Step 2.4: Optimize all parameters using bgc_data and piControl_data
    print("\n=== Step 2.4: Optimizing all parameters using bgc_data and piControl_data ===")
    
    # Define known and unknown parameters for this step
    knowns_dict = {
        'Ksoil_0': Ksoil_0,
        'alpha': alpha,
        'Kresp_0': params['Kresp_0'],
        'Cland_0': params['Cland_0'],
        'Ktfp_tas0': params['Ktfp_tas0'],
        'Ktfp_pr0': params['Ktfp_pr0']
    }
    
    unknowns_dict = {
        'Ktfp_0': [params['Ktfp_0'] * 0.5, params['Ktfp_0'], params['Ktfp_0'] * 2.0],
        'Ktfp_tas1': [-0.4, params['Ktfp_tas1'], 0.4],
        'Ktfp_tas2': [-0.4, params['Ktfp_tas2'], 0.4],
        'Ktfp_pr1': [-0.4, params['Ktfp_pr1'], 0.4],
        'Ktfp_pr2': [-0.4, params['Ktfp_pr2'], 0.4],
        'Ktfp_co2_half': [10.0,  params['Ktfp_co2_half'], 10000.0]
    }
    
    # Run multi-DataFrame optimization
    optimal_params = model_instance.optimize_parameters_multi(
        knowns=knowns_dict,
        unknowns=unknowns_dict,
        data_dfs=[bgc_data, piControl_data],
        co2_dfs=[co2_data, None]  # CO2 data for bgc_data, None for piControl_data
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.4 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Collect Step 2.4 results
    step2_4_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_half']}
    
    # Add to parameter dictionary
    all_parameter_results[(region, model, 'step2_4')] = params.copy()
    

    
    # Step 2.5: Final optimization of climate sensitivity parameters using all data
    print("\n=== Step 2.5: Final optimization of climate sensitivity parameters using all data ===")
    
    # Define known and unknown parameters for this step
    knowns_dict = {
        'Ksoil_0': Ksoil_0,
        'alpha': alpha,
        'Kresp_0': params['Kresp_0'],
        'Cland_0': params['Cland_0'],
        'Ktfp_0': params['Ktfp_0'],
        'Ktfp_tas0': params['Ktfp_tas0'],
        'Ktfp_pr0': params['Ktfp_pr0'],
        'Ktfp_co2_half': params['Ktfp_co2_half']
    }
    
    unknowns_dict = {
        'Ktfp_tas1': [-0.4, params['Ktfp_tas1'], 0.4],
        'Ktfp_tas2': [-0.4, params['Ktfp_tas2'], 0.4],
        'Ktfp_pr1': [-0.4, params['Ktfp_pr1'], 0.4],
        'Ktfp_pr2': [-0.4, params['Ktfp_pr2'], 0.4]
    }
    
    # Run multi-DataFrame optimization
    optimal_params = model_instance.optimize_parameters_multi(
        knowns=knowns_dict,
        unknowns=unknowns_dict,
        data_dfs=[piControl_data, bgc_data, full_data],
        co2_dfs=[None, co2_data, co2_data]  # CO2 data for bgc_data and full_data, None for piControl_data
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.5 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Collect Step 2.5 results
    step2_5_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_half']}
    
    # Add to parameter dictionary
    all_parameter_results[(region, model, 'step2_5')] = params.copy()
    

    
    # Step 2.6: Complete optimization using all datasets (formerly Step 3)
    print("\n=== Step 2.6: Complete optimization using all datasets ===")
    
    # Define known and unknown parameters for complete optimization
    knowns_dict = {
        'Ksoil_0': Ksoil_0,
        'alpha': alpha,
        'Kresp_0': params['Kresp_0'],
        'Cland_0': params['Cland_0'],
        'Ktfp_tas0': params['Ktfp_tas0'],
        'Ktfp_pr0': params['Ktfp_pr0']
    }
    
    # Define unknowns dictionary with explicit bounds for each parameter
    unknowns_dict = {
        'Ktfp_0': [params['Ktfp_0'] * 0.5, params['Ktfp_0'], params['Ktfp_0'] * 2.0],
        'Ktfp_tas1': [-0.4, params['Ktfp_tas1'], 0.4],
        'Ktfp_tas2': [-0.4, params['Ktfp_tas2'], 0.4],
        'Ktfp_pr1': [-0.4, params['Ktfp_pr1'], 0.4],
        'Ktfp_pr2': [-0.4, params['Ktfp_pr2'], 0.4],
        'Ktfp_co2_half': [10.0, params['Ktfp_co2_half'], 10000.0]
    }
    
    # Run multi-DataFrame optimization using all datasets
    optimal_params = model_instance.optimize_parameters_multi(
        knowns=knowns_dict,
        unknowns=unknowns_dict,
        data_dfs=[piControl_data, bgc_data, full_data],
        co2_dfs=[None, co2_data, co2_data]  # CO2 data for bgc_data and full_data, None for piControl_data
    )
    
    # Update parameters with optimized values
    for param, value in optimal_params.items():
        params[param] = value
    
    print(f"Step 2.6 optimization results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Collect Step 2.6 results
    step2_6_params = {param: params[param] for param in ['Kresp_0', 'Cland_0', 'Ktfp_0', 'Ktfp_tas0', 'Ktfp_tas1', 'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2', 'Ktfp_co2_half']}
    
    # Add to parameter dictionary
    all_parameter_results[(region, model, 'step2_6')] = params.copy()
    

    
    return params, all_parameter_results





def run_all_simulations(optimization_results: Dict[str, Dict[str, float]], regions: List[str], models: List[str], all_parameter_results: Dict[Tuple[str, str, str], Dict[str, float]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run all forward simulations for all regions/models and all steps.
    
    Args:
        optimization_results: Dictionary of optimization results for each region/model
        regions: List of regions
        models: List of models
        all_parameter_results: Pre-built dictionary with (region, model, step) keys containing complete parameter sets
        
    Returns:
        Dictionary of simulation results organized by region_model -> step -> DataFrame
    """
    print("=== Running All Forward Simulations ===")
    
    # Run simulations using the pre-built parameter dictionary
    simulation_results = {}
    
    for region in regions:
        for model in models:
            region_model_key = f"{region}_{model}"
            print(f"\nProcessing simulations for: {region} / {model}")
            
            # Load data for this region/model
            piControl_data, full_data, bgc_data = load_data_for_analysis([region], [model])
            co2_data = load_co2_data()
            
            # Initialize simulation results for this region/model
            simulation_results[region_model_key] = {}
            
            # Create model instance
            model_instance = CoinBGC()
            
            # Run simulations for each step using the exact parameters
            steps_to_run = ['step2_1', 'step2_2', 'step2_3', 'step2_4', 'step2_5', 'step2_6', 'piControl', 'full', 'bgc']
            
            for step in steps_to_run:
                param_key = (region, model, step)
                if param_key not in all_parameter_results:
                    print(f"  Warning: No parameters found for {region}/{model}/{step}")
                    continue
                
                print(f"  Running {step} simulation...")
                params = all_parameter_results[param_key]
                
                # Choose appropriate data and run simulation
                if step in ['step2_1', 'step2_2', 'piControl']:
                    sim = model_instance.execute_model(piControl_data, params)
                elif step == 'step2_3':
                    sim = model_instance.execute_model(bgc_data, params, co2_data)
                elif step in ['step2_4', 'step2_5', 'step2_6', 'full', 'bgc']:
                    if step == 'bgc':
                        sim = model_instance.execute_model(bgc_data, params, co2_data)
                    else:
                        sim = model_instance.execute_model(full_data, params, co2_data)
                
                sim['region'] = region
                sim['model'] = model
                simulation_results[region_model_key][step] = sim
    
    return simulation_results


def save_consolidated_parameter_files(all_parameter_results: Dict[Tuple[str, str, str], Dict[str, float]], optimization_results: Dict[str, Dict[str, float]]):
    """
    Save consolidated parameter files for each step containing all regions/models.
    
    Args:
        all_parameter_results: Dictionary with (region, model, step) keys containing complete parameter sets
        optimization_results: Dictionary of final optimization results for each region/model
    """
    output_dir = get_run_output_directory()
    timestamp = _current_run_timestamp
    
    # Get all unique steps
    all_steps = set()
    for (region, model, step) in all_parameter_results.keys():
        all_steps.add(step)
    
    # Create consolidated file for each step
    for step_name in sorted(all_steps):
        consolidated_data = []
        
        for (region, model, step), params in all_parameter_results.items():
            if step == step_name:
                row_data = {'region': region, 'model': model}
                row_data.update(params)
                consolidated_data.append(row_data)
        
        if consolidated_data:
            # Create DataFrame and save
            df = pd.DataFrame(consolidated_data)
            filename = f"fitted_parameters_all_{step_name}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"  Saved: {filename}")


def generate_all_outputs(simulation_results: Dict[str, Dict[str, pd.DataFrame]], optimization_results: Dict[str, Dict[str, float]], all_parameter_results: Dict[Tuple[str, str, str], Dict[str, float]]):
    """
    Generate all CSV files and PDF books from simulation results.
    
    Args:
        simulation_results: Dictionary of simulation results from run_all_simulations
        optimization_results: Dictionary of optimization results from run_main_analysis
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
    
    # Generate consolidated parameter files
    print("Generating consolidated parameter files...")
    save_consolidated_parameter_files(all_parameter_results, optimization_results)
    
    # Generate PDF books
    print("Generating PDF books...")
    create_pdf_books_from_simulation_results(simulation_results, optimization_results, all_parameter_results)


def create_pdf_books_from_simulation_results(simulation_results: Dict[str, Dict[str, pd.DataFrame]], optimization_results: Dict[str, Dict[str, float]], all_parameter_results: Dict[Tuple[str, str, str], Dict[str, float]] = None):
    """
    Create PDF books from simulation results.
    
    Args:
        simulation_results: Dictionary of simulation results from run_all_simulations
        optimization_results: Dictionary of optimization results from run_main_analysis
    """
    print("Creating PDF books from simulation results...")
    
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
                    
                    # Set y-axis bounds ensuring 0 is on the chart
                    all_data_values = list(sim_df['gpp_data']) + list(sim_df['GPP_model'])
                    set_y_axis_bounds(ax, all_data_values)
                    
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
                    
                    # Add parameter information from actual simulation data
                    try:
                        # Extract parameters from the actual simulation data
                        param_columns = ['Ksoil_0', 'Kresp_0', 'Ktfp_0', 'alpha', 'Cland_0', 
                                        'Ktfp_co2_half', 'Ktfp_tas0', 'Ktfp_tas1', 
                                        'Ktfp_tas2', 'Ktfp_pr0', 'Ktfp_pr1', 'Ktfp_pr2']
                        
                        # Get parameters from the first row (they should be constant throughout simulation)
                        first_row = sim_df.iloc[0]
                        step_params = {}
                        for param in param_columns:
                            if param in first_row:
                                step_params[param] = first_row[param]
                        
                        # Create parameter text
                        param_text = "Parameters (from simulation):\n"
                        for param_name, param_value in step_params.items():
                            param_text += f"  {param_name}: {param_value:.6f}\n"
                        
                        # Add parameter text box
                        ax.text(0.02, 0.02, param_text, transform=ax.transAxes,
                               verticalalignment='bottom', fontsize=8, fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    except Exception as e:
                        # If parameter extraction fails, just continue without parameters
                        print(f"Warning: Could not extract parameters from simulation data for {step_name} {region} {model}: {e}")
                
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
                
                # Set y-axis bounds ensuring 0 is on the chart
                all_data_values = (list(df_bgc['gpp_data']) + list(df_bgc['GPP_model']) + 
                                 list(df_full['gpp_data']) + list(df_full['GPP_model']))
                set_y_axis_bounds(ax, all_data_values)
                
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


def run_main_analysis(regions: List[str], models: List[str], Ksoil_0: float, alpha: float, max_iterations: int = 100000) -> Dict[str, Dict[str, float]]:
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
    all_parameter_results = {}  # Collect all parameter dictionaries
    
    # Phase 1: Run all optimizations (save only parameters)
    for region in regions:
        for model in models:
            print(f"\n{'='*60}")
            print(f"Processing: {region} / {model}")
            print(f"{'='*60}")
            
            # Step 1: Read in data
            print("\nStep 1: Loading data...")
            # Load filtered data for simulations
            piControl_data, full_data, bgc_data = load_data_for_analysis([region], [model])
            co2_data = load_co2_data()
            
            print(f"Loaded data:")
            print(f"  piControl_data: {len(piControl_data)} rows")
            print(f"  full_data: {len(full_data)} rows")
            print(f"  bgc_data: {len(bgc_data)} rows")
            print(f"  co2_data: {len(co2_data)} rows")
            
            # Step 2: Run all optimizations
            print("\nStep 2: Running all optimizations...")
            final_params, region_parameter_results = run_optimizations(
                piControl_data, full_data, bgc_data, co2_data, Ksoil_0, alpha, region, model
            )
            
            # Store results
            results[f"{region}_{model}"] = final_params
            
            # Add final parameters to the parameter dictionary
            region_parameter_results[(region, model, 'piControl')] = final_params.copy()
            region_parameter_results[(region, model, 'full')] = final_params.copy()
            region_parameter_results[(region, model, 'bgc')] = final_params.copy()
            
            # Merge into the global parameter dictionary
            all_parameter_results.update(region_parameter_results)
            
            print(f"\nFinal parameters for {region} / {model}:")
            for param, value in final_params.items():
                print(f"  {param}: {value:.6f}")
    
    # Save final optimization results
    print(f"\n=== Saving Optimization Results ===")
    save_fitted_parameters(results, "complete")
    
    # Phase 2: Run all forward simulations
    print(f"\n=== Phase 2: Running All Forward Simulations ===")
    simulation_results = run_all_simulations(results, regions, models, all_parameter_results)
    
    # Phase 3: Generate all outputs (CSV files and PDF books)
    print(f"\n=== Phase 3: Generating All Outputs ===")
    generate_all_outputs(simulation_results, results, all_parameter_results)
    
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
    unknowns = ['Ktfp_0', 'Kresp_0', 'Ktfp_co2_half']  # System optimizes these
    
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
        'Ktfp_co2_half': 10.0
    }
    
    # Example 1: Basic model execution
    print("=== Example 1: Basic Model Execution ===")
    # Create complete parameter dictionary for example
    complete_params = model.create_parameter_dict(known_values)
    results = model.execute_model(sample_data, complete_params)
    print("Model execution results:")
    print(results[['year', 'gpp_data', 'GPP_model', 'Cland']].head())
    
    # Example 2: Parameter optimization
    print("\n=== Example 2: Parameter Optimization ===")
    # Define known and unknown parameters for example
    example_knowns = {
        'Ksoil_0': 0.1,
        'alpha': 0.5,
        'Cland_0': 100.0,
        'Ktfp_co2_half': 10.0
    }
    example_unknowns = {
        'Ktfp_0': [2.0, 1.0, 5.0],
        'Kresp_0': [0.3, 0.1, 0.9]
    }
    optimal_unknowns = model.optimize_parameters(example_knowns, example_unknowns, sample_data)
    print("Optimal unknown parameters:")
    for param, value in optimal_unknowns.items():
        print(f"  {param}: {value:.6f}")
    
    # Example 3: Run model with optimized parameters
    print("\n=== Example 3: Model with Optimized Parameters ===")
    # Create complete parameter dictionary with optimized values
    optimized_params = model.create_parameter_dict(known_values, optimal_unknowns)
    optimized_results = model.execute_model(sample_data, optimized_params)
    print("Optimized model results:")
    print(optimized_results[['year', 'gpp_data', 'GPP_model', 'Cland']].head())


def set_y_axis_bounds(ax, data_values):
    """
    Set y-axis bounds ensuring 0 is always on the chart with appropriate padding.
    
    Args:
        ax: matplotlib axis object
        data_values: list or array of data values to plot
    """
    if not data_values:
        return
    
    min_val = min(data_values)
    max_val = max(data_values)
    
    if min_val >= 0:  # All positive data
        ax.set_ylim(0, 1.1 * max_val)
    elif max_val <= 0:  # All negative data
        ax.set_ylim(1.1 * min_val, 0)
    else:  # Mixed positive and negative data
        ax.set_ylim(1.1 * min_val, 1.1 * max_val)


if __name__ == "__main__":
    example_usage()
