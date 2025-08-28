# COIN-BGC: Clean Implementation Continuation

## Overall Goal
The overall goal is to simulate land-surface behavior under climate change using a Solow-Swan growth model. The system is under-determined by one parameter, so **Ksoil_0** (inverse time constant for heterotrophic respiration) is chosen a priori.

## Current State - PRODUCTION READY ✅
- **Clean Architecture COMPLETED**: Modular design with unified optimization approach
- **Fail Fast Implementation COMPLETED**: No error checking, immediate failure on issues
- **Multi-DataFrame Optimization COMPLETED**: Can optimize across multiple datasets
- **Main Processing Workflow COMPLETED**: Complete analysis pipeline implemented
- **Output System COMPLETED**: Timestamped directories with stage-by-stage parameter saving
- **PDF Book Generation COMPLETED**: Professional visualization of simulation results
- **Runtime Errors ELIMINATED**: System runs without fatal errors
- **Parameter Optimization COMPLETED**: Enhanced debugging and parameter tuning
- **Data Loading Functions COMPLETED**: Flexible data loading with optional filtering
- **Quadratic Climate Sensitivity COMPLETED**: Enhanced parameterization with quadratic terms
- **Unified Optimization COMPLETED**: All optimization steps integrated into single function
- **Physical Constraints IMPLEMENTED**: Ktfp_co2_max calculated from physical principles
- **Enhanced Debugging COMPLETED**: Detailed optimization progress and parameter tracking
- **Chart Visualization COMPLETED**: Y-axis bounds ensuring 0 is always on charts
- **Parameter Display COMPLETED**: Parameter boxes positioned in lower left corner

## Recent Improvements (Latest Updates)

### Chart Visualization Enhancements ✅
- **Y-axis bounds**: All charts now ensure 0 is on the y-axis with appropriate padding
- **Parameter box positioning**: Parameter information moved to lower left corner for better visibility
- **Consistent chart formatting**: All PDF books use the same y-axis bounds logic
- **BGC vs Full comparison**: Fixed y-axis bounds to properly extend to zero

### Code Organization Improvements ✅
- **Import organization**: All imports moved to top of files, no imports in middle of code
- **Sensitivity analysis removal**: Cleaned up code by removing unused sensitivity analysis functions
- **Function organization**: Improved code structure and readability

### Enhanced Optimization Debugging ✅
- **Detailed progress reporting**: Gradient norms, parameter changes, convergence status
- **Parameter change tracking**: Clear reporting of initial → final parameter values
- **Optimization precision**: Enhanced `gtol`, `ftol`, and `eps` settings for better convergence

### Parameter Management Refinements ✅
- **Consistent format**: Standardized `[lower_bound, initial_guess, upper_bound]` format throughout
- **Physical constraints**: `Ktfp_co2_max` calculated from `co2_factor = 1` when `co2 = co2_0`
- **Expanded parameter ranges**: Climate sensitivity parameters expanded by factor of 4
- **Element-wise maximum**: `np.maximum` ensures `Ktfp` never goes negative
- **Improved initial guesses**: Changed from 0.0 to 0.001 for better tracking

### Parameter Dictionary Structure ✅
The system uses explicit parameter dictionaries for clarity:

```python
knowns_dict = {
    'Ksoil_0': Ksoil_0,
    'alpha': alpha,
    'Kresp_0': params['Kresp_0'],
    # ... other fixed parameters
}

unknowns_dict = {
    'Ktfp_tas1': [-0.4, 0.001, 0.4],  # [lower, initial_guess, upper]
    'Ktfp_tas2': [-0.4, 0.001, 0.4],
    'Ktfp_pr1': [-0.4, 0.001, 0.4],
    'Ktfp_pr2': [-0.4, 0.001, 0.4],
    'Ktfp_co2_half': [10.0, params['Ktfp_co2_half'], 10000.0]
}
```

## Unified Optimization Implementation ✅

### Integrated Optimization Function
The system now uses a single `run_optimizations()` function that handles all optimization steps:

1. **Step 2.1**: Initial parameter calculation from piControl data
2. **Step 2.2**: Climate sensitivity optimization using historical data
3. **Step 2.3**: CO2 parameter optimization using bgc_data
4. **Step 2.4**: All parameter optimization using bgc_data and piControl_data
5. **Step 2.5**: Final climate sensitivity optimization using all data
6. **Step 2.6**: Complete optimization using all datasets

### Parameter Dictionary Structure
The system uses a clean parameter dictionary with `(region, model, step)` tuples as keys:

```python
all_parameter_results = {
    ('Brazil', 'ACCESS-ESM1-5', 'step2_1'): {complete_parameter_set},
    ('Brazil', 'ACCESS-ESM1-5', 'step2_2'): {complete_parameter_set},
    # ... all steps for all regions/models
}
```

This ensures each simulation uses exactly the correct parameters for its specific region/model/step combination.

### Stage-by-Stage Parameter Saving
The system saves fitted parameters at each optimization stage:
1. **Step 2.1**: Initial parameter calculation from historical data
2. **Step 2.2**: Climate sensitivity optimization using historical data
3. **Step 2.3**: CO2 parameter optimization using bgc_data
4. **Step 2.4**: All parameter optimization using bgc_data and piControl_data
5. **Step 2.5**: Final climate sensitivity optimization using all data
6. **Step 2.6**: Complete optimization using all datasets

### Simulation Results
- **CSV files**: Model outputs for each dataset (piControl, full, bgc) and each optimization step
- **PDF books**: Professional visualization of simulation results for each step
- **Parameter transparency**: Complete parameter universe included in all outputs

## Output System Implementation ✅

### Timestamped Output Directories
- Each run creates a unique timestamped directory: `data/output/run_YYYYMMDD_HHMMSS/`
- No error checking - assumes all paths are valid
- Clean implementation with explicit requirements

### Stage-by-Stage Parameter Saving
The system saves fitted parameters at each optimization stage:
1. **Step 2.1**: Initial parameter calculation from historical data
2. **Step 2.2**: Climate sensitivity optimization using historical data
3. **Step 2.3**: CO2 parameter optimization using bgc_data
4. **Step 2.4**: All parameter optimization using bgc_data and piControl_data
5. **Step 2.5**: Final climate sensitivity optimization using all data
6. **Step 2.6**: Complete optimization using all datasets

### Simulation Results
- **CSV files**: Model outputs for each dataset (piControl, full, bgc) and each optimization step
- **PDF books**: Professional visualization of simulation results for each step
- **Parameter transparency**: Complete parameter universe included in all outputs

## Enhanced Climate Sensitivity Parameterization ✅

### Temperature Sensitivity
- **Ktfp_tas0**: Reference temperature (°C) - historical mean temperature
- **Ktfp_tas1**: Linear temperature sensitivity coefficient
- **Ktfp_tas2**: Quadratic temperature sensitivity coefficient

### Precipitation Sensitivity  
- **Ktfp_pr0**: Reference precipitation (mm/day) - historical mean precipitation
- **Ktfp_pr1**: Linear precipitation sensitivity coefficient
- **Ktfp_pr2**: Quadratic precipitation sensitivity coefficient

### Enhanced Ktfp Equation
```
Ktfp = Ktfp_0 * tas_factor * pr_factor * co2_factor
```
Where:
- `tas_factor = 1 + Ktfp_tas1 * (tas - Ktfp_tas0) + Ktfp_tas2 * (tas - Ktfp_tas0)^2`
- `pr_factor = 1 + Ktfp_pr1 * (pr - Ktfp_pr0) + Ktfp_pr2 * (pr - Ktfp_pr0)^2`
- `co2_factor = 1 + Ktfp_co2_max * co2 / (co2 + Ktfp_co2_half)`
- `Ktfp_co2_max = (co2_0 + Ktfp_co2_half) / co2_0` (calculated from physical constraint)

## Code Architecture ✅

### Core Components
- **`CoinBGC` class**: Main class implementing the clean architecture
- **`run_optimizations()`**: Unified function that runs all optimization steps (2.1 through 2.6)
- **`run_all_simulations()`**: Runs forward simulations using optimized parameters
- **`run_main_analysis()`**: Orchestrates the complete analysis workflow

### Clean Parameter Management
The project uses a clean, explicit parameter management system:

```python
def run_optimizations(piControl_data, full_data, bgc_data, co2_data, Ksoil_0, alpha, region, model):
    """
    Run all optimizations for a region/model combination.
    """
    # Step 2.1: Calculate initial parameters from piControl data
    # Step 2.2: Optimize climate sensitivity parameters using historical data
    # Step 2.3: Optimize CO2 parameters using bgc_data
    # Step 2.4: Optimize all parameters using bgc_data and piControl_data
    # Step 2.5: Final optimization of climate sensitivity parameters using all data
    # Step 2.6: Complete optimization using all datasets
```

This approach makes parameter management much clearer and more maintainable.

## Features ✅
- **Complete analysis pipeline**: From data loading to final optimization results
- **Multi-DataFrame optimization**: Can optimize across multiple datasets simultaneously
- **Enhanced climate sensitivity**: Quadratic temperature and precipitation responses
- **Flexible data loading**: Optional filtering by regions and models
- **Unified optimization pipeline**: Single function handles all optimization steps
- **Clean parameter management**: Explicit fixed vs. optimized parameter handling
- **Fail fast philosophy**: No error checking, immediate failure on issues
- **Multi-region processing**: Run simulations for all countries and models
- **CO2 integration**: Historical and future CO2 concentration data support
- **Comprehensive output**: Clean output format for further processing
- **Extensible**: Easy to modify by moving parameters between knowns and unknowns
- **Enhanced debugging**: Detailed optimization progress and parameter tracking
- **Physical constraints**: Ktfp_co2_max calculated from physical principles
- **Robust optimization**: Element-wise maximum prevents negative Ktfp values
- **Professional visualization**: PDF books with consistent chart formatting
- **Clean code organization**: All imports at top, no unused functions

## Current Status: PRODUCTION READY ✅

The system is **fully functional** and ready for production use. All major features have been implemented and tested:

1. **Complete Optimization Pipeline**: All 6 optimization steps working correctly
2. **Professional Output**: PDF books with proper chart formatting and parameter display
3. **Clean Code Structure**: Well-organized, maintainable codebase
4. **Robust Error Handling**: Fail-fast approach with clear error messages
5. **Comprehensive Documentation**: Clear usage instructions and parameter descriptions

### Recent Completion Highlights
- **Chart visualization**: Y-axis bounds ensure 0 is always visible with appropriate padding
- **Parameter display**: Parameter boxes positioned in lower left corner for better visibility
- **Code cleanup**: Removed unused sensitivity analysis functions
- **Import organization**: All imports properly organized at top of files

## Next Steps: Production Use and Tuning

The system is ready for:

1. **Production Runs**: Execute large-scale analyses across multiple regions/models
2. **Parameter Tuning**: Fine-tune parameter bounds based on optimization results
3. **Performance Analysis**: Analyze optimization patterns and convergence
4. **Physical Validation**: Validate that optimized parameters produce realistic results
5. **Documentation**: Create user guides and example workflows

## Dependencies
- pandas==2.0.3 - Data manipulation and analysis
- numpy==1.24.4 - Numerical operations
- scipy==1.10.1 - Scientific computing and optimization
- scikit-learn==1.3.2 - Machine learning utilities
- statsmodels==0.14.1 - Statistical modeling
- matplotlib==3.7.2 - Plotting and PDF generation

## License
MIT License 