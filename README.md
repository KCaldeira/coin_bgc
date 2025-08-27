# COIN-BGC: Solow-Swan Growth Model for Land-Surface Climate Change Simulation

**⚠️ PROTOTYPE STATUS: This is a prototype version of the COIN-BGC system. The current codebase represents a working proof-of-concept that demonstrates the core functionality. A clean, more general version has been implemented in `coin_bgc.py` with improved architecture, design principles, and a complete main processing workflow.**

This project simulates the behavior of the land-surface model under climate change using a Solow-Swan growth model of an economy. The model represents terrestrial carbon cycling as an economic system where carbon stocks (Cland) are the "capital" that produces carbon fluxes (GPP, NPP) through biological processes.

## Prototype to Production: Next Version Design Principles

The next version of COIN-BGC will be built with the following design principles:

### More General and More Specific
- **More General**: Basic routines will be made as general as possible for reusability
- **More Specific**: Code will be written for one specific use case, but designed for easy adaptation to other use cases
- **No Conditional Clutter**: Avoid cluttering code with conditional statements
- **Clean Code Philosophy**: Minimal conditionals, explicit parameter requirements, no hidden assumptions

### Central Design: Three Lists of Keys
The new version will be built around three fundamental lists of keys:

1. **`knowns`** - Variables to be specified in an optimization (user-provided parameters)
2. **`unknowns`** - Variables to be optimized for (parameters to be determined)
3. **`universe`** - Complete set of variables (all possible parameters)

### Core Architecture
The clean version implements two fundamental routines:

1. **Basic Model Execution Routine**: 
   - Executes the model from start year to end year
   - Takes a pandas DataFrame and a dictionary of parameter values for the set of knowns
   - Sets all variables not in `knowns` (i.e., `universe - knowns`) to 0.0

2. **Optimization Routine**:
   - Optimizes for the `unknowns` given the `knowns`
   - Takes a pandas DataFrame and the known parameter values
   - Returns optimal values for the unknown parameters

This architecture provides a clean, modular foundation that can be easily adapted to different use cases without extensive code modifications.

### Clean Implementation: `coin_bgc.py`
The clean architecture has been implemented in `coin_bgc.py` with the following features:

- **`CoinBGC` class**: Main class implementing the clean architecture
- **Three lists of keys**: `knowns`, `unknowns`, and `universe` for parameter management
- **`execute_model()`**: Basic model execution routine
- **`optimize_parameters()`**: Single DataFrame optimization routine
- **`optimize_parameters_multi()`**: Multi-DataFrame optimization routine
- **Clean parameter management**: No conditional clutter, explicit parameter handling
- **Data loading functions**: Flexible data loading with optional filtering
- **Main processing workflow**: Complete analysis pipeline with preliminary and final optimizations
- **Quadratic climate sensitivity**: Enhanced temperature and precipitation parameterization
- **Example usage**: Complete working example demonstrating the architecture

## Overall Goal
The overall goal of this effort is to simulate the behavior of the land-surface model under climate change using a Solow-Swan growth model of an economy. The system is under-determined by one parameter, so one parameter needs to be chosen a priori. The parameter that we will choose to specify is **Ksoil_0**, which is the inverse time constant for loss of carbon stock due to heterotrophic respiration.

## Model Framework
The model treats terrestrial carbon cycling as an economic system:
- **Carbon stocks (Cland)** = "Capital" in the economic model
- **Gross Primary Production (GPP)** = "Output" produced by the capital
- **Net Primary Production (NPP)** = "Net output" after respiration costs
- **Soil respiration** = "Depreciation" of the carbon capital

## Model Equations
- **GPP = Ktfp * Cland^alpha** (Production function)
- **NPP = (1 - Kresp) * GPP** (Net production after plant respiration)
- **SOILresp = Ksoil * Cland** (Soil respiration/depreciation)
- **dCland/dt = NPP - SOILresp** (Capital accumulation equation)
- **Cland(t+1) = Cland(t) + dCland/dt** (Time stepping)

Where Ktfp can be a function of temperature (tas), precipitation (pr), and CO2 concentration.

## Main Processing Workflow

### Step 1: Data Loading
- **Flexible data loading**: `load_data_for_analysis()` with optional region/model filtering
- **Three datasets**: piControl_data, full_data (historical + SSP585), bgc_data (historical-bgc + SSP585-bgc)
- **CO2 data loading**: `load_co2_data()` for historical and future CO2 concentrations
- **No error checking**: Fail fast if data issues exist

### Step 2: Preliminary Optimizations
**Step 2.1**: Calculate initial parameters from historical data means
- `Kresp_0 = npp_mean / gpp_mean`
- `Cland_0 = npp_mean / Ksoil_0`
- `Ktfp_0 = gpp_mean / (Cland_0 ** alpha)`

**Step 2.2**: Set reference values to historical means
- `Ktfp_tas0 = tas_mean`
- `Ktfp_pr0 = pr_mean`

**Step 2.3**: Optimize climate sensitivity parameters using historical data
- Optimizes: `Cland_0`, `Ktfp_0`, `Ktfp_tas1`, `Ktfp_tas2`, `Ktfp_pr1`, `Ktfp_pr2`
- Uses bounds: half to twice the initial values

**Step 2.4**: Optimize CO2 parameters using bgc_data
- Optimizes: `Ktfp_co2_max`, `Ktfp_co2`
- Uses fixed bounds for CO2 parameters

**Step 2.5**: Optimize all parameters using bgc_data and piControl_data
- Uses multi-DataFrame optimization
- Optimizes all parameters except command line ones

**Step 2.6**: Final optimization of climate sensitivity using all data
- Uses multi-DataFrame optimization with all three datasets
- Optimizes only: `Ktfp_tas1`, `Ktfp_tas2`, `Ktfp_pr1`, `Ktfp_pr2`

### Step 3: Complete Optimization
- Takes preliminary results as starting points
- Uses bounds: half to twice the preliminary values
- Optimizes all parameters using all datasets
- Only keeps command line parameters (`Ksoil_0`, `alpha`) as knowns

### Step 4: Results and Reporting
- Comprehensive parameter results for each region/model combination
- Ready for visualization and analysis
- Clean output format for further processing

## Enhanced Climate Sensitivity Parameterization

The project now uses an advanced climate sensitivity parameterization that includes quadratic terms:

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
- `co2_factor = 1 + Ktfp_co2_max * co2 / (co2 + Ktfp_co2)`

### Physical Interpretation
This enhanced approach provides:
- **Linear and quadratic responses**: Captures both linear and non-linear climate effects
- **Better fit to data**: More flexible parameterization for complex climate responses
- **Physical meaning**: Maintains interpretable sensitivity coefficients
- **Robust optimization**: Better convergence with enhanced parameter space

## Code Architecture

The project has been refactored into a clean, modular architecture:

- **`coin_bgc.py`**: Complete clean implementation with main processing workflow
  - **`CoinBGC` class**: Main class implementing the clean architecture
  - **`load_data_for_analysis()`**: Flexible data loading with optional filtering
  - **`load_co2_data()`**: CO2 concentration data loading
  - **`run_preliminary_optimizations()`**: Complete Step 2 preliminary optimization pipeline
  - **`run_complete_optimization()`**: Step 3 complete optimization
  - **`run_main_analysis()`**: Main orchestrator for complete analysis workflow
  - **`execute_model()`**: Basic model execution routine
  - **`optimize_parameters()`**: Single DataFrame optimization
  - **`optimize_parameters_multi()`**: Multi-DataFrame optimization
  - **Clean parameter management**: No conditional clutter, explicit parameter handling

### New Clean Parameter Optimization Approach

The project now uses a clean, explicit parameter management system:

```python
def run_preliminary_optimizations(piControl_data, full_data, bgc_data, co2_data, Ksoil_0, alpha):
    """
    Run preliminary optimizations to get starting points for complete optimization.
    """
    # Step 2.1: Calculate initial parameters from historical data means
    # Step 2.2: Set reference values to historical means
    # Step 2.3: Optimize climate sensitivity parameters using historical data
    # Step 2.4: Optimize CO2 parameters using bgc_data
    # Step 2.5: Optimize all parameters using bgc_data and piControl_data
    # Step 2.6: Final optimization of climate sensitivity using all data
```

This approach makes parameter management much clearer and more maintainable.

## Features
- **Complete analysis pipeline**: From data loading to final optimization results
- **Multi-DataFrame optimization**: Can optimize across multiple datasets simultaneously
- **Enhanced climate sensitivity**: Quadratic temperature and precipitation responses
- **Flexible data loading**: Optional filtering by regions and models
- **Preliminary optimization pipeline**: Step-by-step approach for robust starting points
- **Complete optimization pipeline**: Final optimization using all datasets
- **Clean parameter management**: Explicit fixed vs. optimized parameter handling
- **Fail fast philosophy**: No error checking, immediate failure on issues
- **Multi-region processing**: Run simulations for all countries and models
- **CO2 integration**: Historical and future CO2 concentration data support
- **Comprehensive output**: Clean output format for further processing
- **Extensible**: Easy to modify by moving parameters between knowns and unknowns

## Getting Started

### 1. Setup Virtual Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Analysis
```python
# Import the main analysis function
from coin_bgc import run_main_analysis

# Run complete analysis
regions = ["Zimbabwe", "China"]
models = ["ACCESS-ESM1-5"]
Ksoil_0 = 0.1
alpha = 0.5

results = run_main_analysis(regions, models, Ksoil_0, alpha)

# Access results
for key, params in results.items():
    print(f"Results for {key}:")
    for param, value in params.items():
        print(f"  {param}: {value:.6f}")
```

### 3. Run Individual Components
```python
# Load data
from coin_bgc import load_data_for_analysis, load_co2_data

piControl_data, full_data, bgc_data = load_data_for_analysis(["Zimbabwe"], ["ACCESS-ESM1-5"])
co2_data = load_co2_data()

# Run preliminary optimizations
from coin_bgc import run_preliminary_optimizations

preliminary_params = run_preliminary_optimizations(
    piControl_data, full_data, bgc_data, co2_data, Ksoil_0=0.1, alpha=0.5
)

# Run complete optimization
from coin_bgc import run_complete_optimization

final_params = run_complete_optimization(
    piControl_data, full_data, bgc_data, co2_data, preliminary_params
)
```

### 4. Use the CoinBGC Class Directly
```python
from coin_bgc import CoinBGC

# Initialize the system
model = CoinBGC()

# Set parameter sets
knowns = ['Ksoil_0', 'alpha']
unknowns = ['Ktfp_0', 'Kresp_0', 'Cland_0']
model.set_parameter_sets(knowns, unknowns)

# Run model execution
results = model.execute_model(data_df, known_values)

# Run optimization
optimal_params = model.optimize_parameters(known_values, data_df, initial_guesses, bounds)
```

## Input Data
Place your input CSV files in `data/input/`. The project uses:
- `Data_regression_piControl.csv` - Pre-industrial control data
- `Data_regression_historical.csv` - Historical data
- `Data_regression_ssp585.csv` - SSP585 scenario data
- `Data_regression_hist-bgc.csv` - Historical biogeochemical data
- `Data_regression_ssp585-bgc.csv` - SSP585 biogeochemical scenario data
- `historical-ssp585_co2.csv` - Historical and future CO2 concentrations

## Parameter Universe

The complete parameter universe includes:

- **Ksoil_0**: Inverse time constant for soil respiration
- **Kresp_0**: Plant respiration fraction
- **Ktfp_0**: Total factor productivity (base)
- **alpha**: Production function exponent
- **Cland_0**: Initial carbon land stock
- **Ktfp_co2**: CO2 half-saturation value (ppm)
- **Ktfp_co2_max**: CO2 fertilization maximum factor
- **Ktfp_tas0**: Reference temperature
- **Ktfp_tas1**: Linear temperature sensitivity coefficient
- **Ktfp_tas2**: Quadratic temperature sensitivity coefficient
- **Ktfp_pr0**: Reference precipitation
- **Ktfp_pr1**: Linear precipitation sensitivity coefficient
- **Ktfp_pr2**: Quadratic precipitation sensitivity coefficient

## Current Status
- **Clean Architecture**: ✅ **COMPLETED** - Modular design with three lists of keys
- **Fail Fast Implementation**: ✅ **COMPLETED** - No error checking, immediate failure
- **Multi-DataFrame Optimization**: ✅ **COMPLETED** - Can optimize across multiple datasets
- **Main Processing Workflow**: ✅ **COMPLETED** - Complete analysis pipeline
- **Data Loading Functions**: ✅ **COMPLETED** - Flexible data loading with optional filtering
- **Enhanced Climate Sensitivity**: ✅ **COMPLETED** - Quadratic temperature and precipitation responses
- **Preliminary Optimization Pipeline**: ✅ **COMPLETED** - Step-by-step optimization approach
- **Complete Optimization Pipeline**: ✅ **COMPLETED** - Final optimization using all datasets

## Next Steps: Production Implementation

The next phase will focus on:

1. **Production Testing**: Test the complete workflow with real data
2. **Performance Optimization**: Optimize for large-scale runs
3. **Visualization**: Add comprehensive plotting and reporting
4. **Documentation**: Complete user documentation and examples
5. **Validation**: Cross-validation with existing prototype results

## Dependencies
- pandas==2.0.3 - Data manipulation and analysis
- numpy==1.24.4 - Numerical operations
- scipy==1.10.1 - Scientific computing and optimization
- scikit-learn==1.3.2 - Machine learning utilities
- statsmodels==0.14.1 - Statistical modeling
- matplotlib==3.7.2 - Plotting and PDF generation

## License
MIT License 