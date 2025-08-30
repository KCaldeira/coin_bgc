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
- **Centralized bounds**: Simplified JSON structure with single bounds dictionary (August 2025)

## LATEST IMPROVEMENT: Centralized Bounds Architecture ✅ (August 2025)

### Problem Solved
The original JSON structure had repetitive hardcoded bounds on each optimization step, making maintenance difficult and prone to inconsistencies.

### Solution Implemented ✅
**Centralized Bounds Dictionary**: Created a single `bounds` section in the JSON configuration that defines [low, high] ranges for all optimization parameters once.

#### Key Changes:
1. **Single Bounds Definition**: All parameter bounds defined once at the top level
2. **Simplified Step Structure**: Replaced complex knowns/unknowns dictionaries with:
   - `parameters`: Dictionary containing all parameter values/sources
   - `knowns`: Simple list of parameter names to hold fixed  
   - `unknowns`: Simple list of parameter names to optimize
3. **Starting Values**: Initial optimization values specified directly in the parameters dictionary
4. **Parser Updates**: Modified workflow configuration parser to handle new structure and use centralized bounds

#### Benefits Realized:
- **DRY Principle**: Bounds defined once, used consistently across all steps
- **Easier Maintenance**: Change bounds in one place affects all optimization steps
- **Cleaner JSON**: Much simpler and more readable workflow definitions
- **Less Error-Prone**: No risk of inconsistent bounds between steps
- **Better Separation**: Clear distinction between bounds (constraints) and starting values (initialization)

#### Files Updated:
- `workflow_schema_example.json`: Converted to new centralized bounds format
- `workflow_config.py`: Updated parser to handle new structure
- `coin_bgc.py`: Modified optimization step execution to use centralized bounds
- `main.py`: Updated to pass bounds through call chain

### Status: ✅ COMPLETED AND TESTED
The centralized bounds system is fully operational and has been tested successfully with the standard 6-step optimization pipeline.

## MAJOR ENHANCEMENT COMPLETED: JSON-Based Flexible Workflow System ✅

### ⚠️ IMPORTANT: Legacy Files Protection
**DO NOT MODIFY** the following files - they remain as working production code:
- `coin_bgc_econ.py` - Original production implementation 
- `main_econ.py` - Original production main function

The new flexible JSON-driven system uses:
- `coin_bgc.py` - Evolved from coin_bgc_econ.py with JSON workflow implementation
- `main.py` - Main controller for JSON-driven workflows  
- `workflow_config.py` - Configuration management classes
- `workflow_schema_example.json` - Standard 6-step pipeline workflow definition

### Original Limitation (SOLVED ✅)
The original system had a **hardcoded optimization pipeline** with fixed steps (2.1-2.6) where:
- Parameter assignments were hardcoded in `run_optimizations()`
- Dataset usage was fixed per step
- Could not easily modify which parameters are optimized in each step
- Could not reorder steps or create alternative workflows
- No configuration-driven approach

### Solution Implemented: JSON Workflow Configuration ✅ **COMPLETED**
Created a fully flexible system where optimization workflows are defined by JSON configuration files, enabling:

#### ✅ **FULLY IMPLEMENTED**:
1. **JSON Schema Design**: Comprehensive schema in `workflow_schema_example.json`
2. **Configuration Classes**: Complete `workflow_config.py` with:
   - `WorkflowConfig`, `WorkflowStep`, `ParameterSpec` dataclasses
   - `WorkflowConfigLoader` for loading/validating JSON configs
   - `WorkflowExecutor` for resolving parameter values and bounds
3. **Flexible Step Executor**: `coin_bgc.py` executes steps based on JSON configuration via `execute_workflow_step()`
4. **Dynamic Parameter Management**: Config-driven parameter dictionaries with automatic parameter flow
5. **Main Controller**: `main.py` provides three-step execution (load, execute, output) with JSON workflows
6. **Unified Optimization**: Single `optimize_parameters()` method handles both single and multi-dataset cases
7. **Command Line Interface**: Full argument parsing with required user parameters
8. **Integration Testing**: System successfully loads JSON, executes workflows, and performs optimizations

### JSON Configuration Features ✅
- **Global Parameters**: User-input parameters (Ksoil_0, alpha)
- **Step Types**: `calculation`, `optimization` (unified - handles both single and multi-dataset)
- **Parameter Sources**: `global`, `step`, `value` - automatic parameter flow between steps
- **Centralized Bounds**: Single `bounds` dictionary with `[low, high]` ranges, starting values specified per step
- **Data Source Mapping**: Flexible assignment of any datasets to optimization steps
- **Complete Validation**: Built-in validation catches configuration errors early

### Example JSON Structure ✅ - UPDATED: Centralized Bounds
```json
{
  "workflow_name": "COIN-BGC Standard Pipeline",
  "bounds": {
    "Ktfp_0": [0.01, 1.0],
    "Ktfp_tas1": [-0.4, 0.4],
    "Ktfp_tas2": [-0.4, 0.4],
    "Ktfp_pr1": [-0.4, 0.4],
    "Ktfp_pr2": [-0.4, 0.4],
    "Ktfp_co2_half": [10.0, 10000.0]
  },
  "steps": [
    {
      "name": "step2_1",
      "step_type": "calculation", 
      "data_sources": ["piControl"],
      "calculations": {
        "Kresp_0": "1 - npp_mean / gpp_mean",
        "Cland_0": "npp_mean / Ksoil_0"
      }
    },
    {
      "name": "step2_2",
      "step_type": "optimization",
      "data_sources": ["historical"],
      "parameters": {
        "Ksoil_0": {"source": "global"},
        "Kresp_0": {"source": "step", "step": "step2_1"},
        "Ktfp_tas1": 0.001
      },
      "knowns": ["Ksoil_0", "Kresp_0"],
      "unknowns": ["Ktfp_tas1"]
    }
  ]
}
```

### Benefits of JSON Workflow System ✅ **REALIZED**
- **Complete Flexibility**: Define any optimization sequence without code changes
- **Parameter Reuse**: Automatic flow of results from previous steps to subsequent steps
- **Dataset Flexibility**: Use any combination of data sources per optimization step
- **Multiple Workflows**: Easy to create different JSON files for different research approaches
- **Easy Modification**: Change workflows by editing JSON files, not code
- **Validation**: Built-in configuration validation catches errors before execution
- **Self-Documenting**: JSON workflows serve as clear documentation of analysis approach
- **Unified Architecture**: Single optimization method handles all cases (single/multi-dataset)
- **Fail-Fast**: Missing configurations cause immediate, clear error messages

### Files Created and Implementation Status ✅
- **`main.py`**: Complete three-step controller (load, execute, output) ✅
- **`coin_bgc.py`**: Flexible workflow execution with `execute_workflow_step()` method ✅
- **`workflow_config.py`**: Configuration loading, validation, and parameter resolution ✅
- **`workflow_schema_example.json`**: Standard 6-step pipeline with simplified bounds ✅
- **Command line interface**: Full argument parsing with required parameters ✅
- **Unified optimization**: Single `optimize_parameters()` method for all cases ✅

### Current Status: JSON-Driven Flexible Workflow System OPERATIONAL ✅
The system successfully:
- ✅ Loads JSON workflow configurations with validation
- ✅ Executes flexible step sequences with automatic parameter flow
- ✅ Performs real optimizations with 100+ function evaluations
- ✅ Handles both single and multi-dataset optimizations through unified interface
- ✅ Provides clear error messages when configurations are incomplete
- ✅ Maintains original production code integrity (coin_bgc_econ.py, main_econ.py untouched)

## READY FOR PRODUCTION USE WITH PARALLEL EXECUTION ✅

### Latest Enhancement: Multi-Core Parallel Processing System ✅ (August 2025)

#### Problem Solved
Large-scale analyses with many regions/models were time-consuming and vulnerable to single-point failures, not utilizing available multi-core resources efficiently.

#### Solution Implemented ✅
**Complete Parallel Execution System**: Implemented comprehensive multi-core processing with fault tolerance and result management.

#### Key Features Added:
1. **Region Pattern Matching**: New `--region-pattern` flag supports glob patterns like `"[AB]*"`, `"A*"`
2. **Parallel Shell Script**: `run_parallel_regions.sh` orchestrates multiple jobs with automatic load balancing
3. **Robust Error Handling**: Individual failures don't stop other jobs, comprehensive error tracking
4. **Result Concatenation**: `concatenate_results.py` automatically merges outputs from parallel runs
5. **Emergency Controls**: `pkill -f "python main.py"` for rapid termination of all jobs
6. **Resource Management**: Configurable core usage (e.g., 6/8 cores) to leave resources free

#### Files Updated:
- `main.py`: Added `--region-pattern` flag and pattern expansion logic
- `run_parallel_regions.sh`: Complete parallel execution orchestration
- `concatenate_results.py`: Automatic result merging and analysis
- `README.md`: Updated with parallel execution documentation
- `CONTINUATION.md`: Documented new parallel processing capabilities

### Current Production-Ready Features ✅

The system now includes:

1. **Core Architecture**: Complete three-step controller with JSON workflow execution
2. **Unified Optimization**: Single method handles all optimization scenarios  
3. **Configuration Management**: Robust JSON loading, validation, and parameter resolution
4. **Command Line Interface**: Full argument parsing for single and parallel execution
5. **Multi-Core Processing**: Efficient parallel execution with automatic load balancing
6. **Fault Tolerance**: Robust error handling with partial result recovery
7. **Result Management**: Automatic concatenation and analysis of parallel runs
8. **Legacy Code Protection**: Original production files remain untouched
9. **Example Workflows**: Standard 6-step pipeline ready for immediate use
10. **Comprehensive Documentation**: Updated README.md and CONTINUATION.md

### Usage Examples

**Single Instance:**
```bash
python main.py --alpha=0.5 --Ksoil_0=0.025 --region-pattern "[AB]*" --models "ACCESS-ESM1-5" --json workflow_schema_example.json
```

**Parallel Execution (Recommended):**
```bash
# Use 6 cores, leave 2 free for other activities
./run_parallel_regions.sh 6

# Emergency stop if needed
pkill -f "python main.py"
```

**Result Merging:**
```bash
python concatenate_results.py "run_20250830_*"
```

### Benefits Realized

1. **Scalability**: Efficiently utilizes 2-16 cores with automatic load balancing
2. **Fault Tolerance**: Individual region/model failures don't stop entire runs
3. **Resource Control**: Configurable core usage prevents system overload
4. **Complete Results**: Partial results recovered even with some failures
5. **Easy Management**: Simple commands for execution, monitoring, and termination
6. **Production Ready**: Handles large-scale analyses with minimal user intervention

### Next Steps for Future Development

The enhanced system enables:

1. **Large-Scale Production Runs**: Process hundreds of regions/models efficiently
2. **Custom Workflows**: Create specialized JSON configurations for different research questions
3. **Alternative Optimization Strategies**: Test different parameter optimization approaches
4. **Performance Analysis**: Analyze optimization patterns and convergence behavior across regions
5. **Integration**: Add new step types, data sources, or optimization methods
6. **Collaborative Research**: Share and version control workflow definitions
7. **Cloud Deployment**: Scale to larger compute clusters or cloud environments

### Technical Debt Addressed ✅
- **Code Duplication**: Eliminated separate single/multi optimization methods
- **Hardcoded Workflows**: Replaced with flexible JSON configuration system  
- **Parameter Management**: Unified parameter handling with automatic flow
- **Configuration Complexity**: Simplified to clean `[lower, initial, upper]` bounds
- **Maintainability**: Clear separation of concerns between config, execution, and output

## Dependencies
- pandas==2.0.3 - Data manipulation and analysis
- numpy==1.24.4 - Numerical operations
- scipy==1.10.1 - Scientific computing and optimization
- scikit-learn==1.3.2 - Machine learning utilities
- statsmodels==0.14.1 - Statistical modeling
- matplotlib==3.7.2 - Plotting and PDF generation

## License
MIT License 