# COIN-BGC: Clean Implementation

A clean implementation of the COIN-BGC (Carbon-Oxygen-Interactive-Network Biogeochemical Cycle) model for land-surface climate change simulation using a Solow-Swan growth model approach.

## Overview

This implementation provides a clean, modular architecture for parameter optimization and forward simulation of land-surface biogeochemical cycles under climate change scenarios. The system uses a step-by-step optimization approach to calibrate model parameters using multiple datasets.

## Architecture

### Core Components

1. **`CoinBGC` Class**: The main model implementation with optimization capabilities
2. **`run_optimizations()`**: Unified function that runs all optimization steps (2.1 through 2.6)
3. **`run_all_simulations()`**: Runs forward simulations using optimized parameters
4. **`run_main_analysis()`**: Orchestrates the complete analysis workflow

### Optimization Steps

The system uses a unified optimization approach with the following steps:

- **Step 2.1**: Calculate initial parameters from piControl data
- **Step 2.2**: Optimize climate sensitivity parameters using historical data
- **Step 2.3**: Optimize CO2 parameters using bgc_data
- **Step 2.4**: Optimize all parameters using bgc_data and piControl_data
- **Step 2.5**: Final optimization of climate sensitivity parameters using all data
- **Step 2.6**: Complete optimization using all datasets

### Parameter Dictionary Structure

The system uses a clean parameter dictionary structure with `(region, model, step)` tuples as keys:

```python
all_parameter_results = {
    ('Brazil', 'ACCESS-ESM1-5', 'step2_1'): {complete_parameter_set},
    ('Brazil', 'ACCESS-ESM1-5', 'step2_2'): {complete_parameter_set},
    # ... all steps for all regions/models
}
```

This ensures each simulation uses exactly the correct parameters for its specific region/model/step combination.

### Parameter Management

The system uses explicit parameter dictionaries for clarity:

- **`knowns_dict`**: Dictionary of fixed parameters with their values
- **`unknowns_dict`**: Dictionary of optimized parameters with format `[lower_bound, initial_guess, upper_bound]`

## Usage

### Command Line Interface

```bash
# Run analysis for a single region/model
python main.py --regions "Brazil" --models "ACCESS-ESM1-5" --Ksoil_0 0.025 --alpha 0.5

# Run analysis for multiple regions
python main.py --regions "Brazil" "China" --models "ACCESS-ESM1-5" --Ksoil_0 0.025 --alpha 0.5

# List available data
python main.py --list-data
```

### Required Parameters

- `--Ksoil_0`: Soil respiration parameter (inverse time constant for heterotrophic respiration)
- `--alpha`: Production function exponent
- `--regions`: List of regions to analyze
- `--models`: List of climate models to analyze

## Output

The system generates comprehensive outputs including:

- **CSV Files**: Simulation results for each step and region/model combination
- **PDF Books**: Visual reports showing GPP data vs model predictions with professional formatting
- **Parameter Files**: Consolidated parameter values for each optimization step
- **Timestamped Directories**: All outputs organized by run timestamp

### Chart Visualization Features

- **Y-axis bounds**: All charts ensure 0 is on the y-axis with appropriate padding (1.1x max value)
- **Parameter display**: Parameter information boxes positioned in lower left corner for better visibility
- **Consistent formatting**: All PDF books use the same chart formatting logic
- **Professional appearance**: Clean, publication-ready visualizations

## Data Requirements

The system expects the following data files:

- `data/input/Data_regression_piControl.csv`: Pre-industrial control data
- `data/input/Data_regression_full.csv`: Historical + SSP585 data  
- `data/input/Data_regression_bgc.csv`: Historical-bgc + SSP585-bgc data
- `data/input/co2_data.csv`: CO2 concentration data

## Model Parameters

The complete parameter universe includes:

- `Ksoil_0`: Soil respiration parameter (user-specified)
- `alpha`: Production function exponent (user-specified)
- `Kresp_0`: Plant respiration fraction
- `Cland_0`: Initial land carbon stock
- `Ktfp_0`: Total factor productivity baseline
- `Ktfp_tas0`, `Ktfp_tas1`, `Ktfp_tas2`: Temperature sensitivity parameters
- `Ktfp_pr0`, `Ktfp_pr1`, `Ktfp_pr2`: Precipitation sensitivity parameters
- `Ktfp_co2_half`: CO2 half-saturation concentration

### Parameter Constraints

- `Ktfp_co2_max` is calculated from the constraint that `co2_factor = 1` when `co2 = co2_0`:
  ```
  Ktfp_co2_max = (co2_0 + Ktfp_co2_half) / co2_0
  ```

## Current Status: PRODUCTION READY ✅

The system is **fully functional** and ready for production use. All major features have been implemented and tested:

### Recent Improvements

- **Chart visualization**: Y-axis bounds ensure 0 is always visible with appropriate padding
- **Parameter display**: Parameter boxes positioned in lower left corner for better visibility
- **Code organization**: All imports properly organized at top of files
- **Enhanced debugging**: Detailed optimization progress and parameter tracking
- **Improved parameter bounds**: Expanded ranges for climate sensitivity parameters
- **Physical constraint implementation**: `Ktfp_co2_max` calculated from physical principles
- **Consistent parameter format**: Standardized `[lower_bound, initial_guess, upper_bound]` format
- **Element-wise maximum**: Ensures `Ktfp` never goes negative using `np.maximum`

## Clean Implementation Features

- **Fail Fast**: No error checking, immediate failure on issues
- **Modular Design**: Clear separation of optimization, simulation, and output generation
- **Parameter Transparency**: Complete parameter universe included in all outputs
- **Consistent Data Flow**: Single parameter dictionary ensures consistency across all steps
- **Timestamped Outputs**: Organized output structure with unique run identifiers
- **Professional Visualization**: Publication-ready PDF books with consistent formatting
- **Clean Code Structure**: Well-organized, maintainable codebase

## Next Steps

The system is ready for:

1. **Production Runs**: Execute large-scale analyses across multiple regions/models
2. **Parameter Tuning**: Fine-tune parameter bounds based on optimization results
3. **Performance Analysis**: Analyze optimization patterns and convergence
4. **Physical Validation**: Validate that optimized parameters produce realistic results
5. **Documentation**: Create user guides and example workflows 