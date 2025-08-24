# COIN-BGC: Solow-Swan Growth Model for Land-Surface Climate Change Simulation

This project simulates the behavior of the land-surface model under climate change using a Solow-Swan growth model of an economy. The model represents terrestrial carbon cycling as an economic system where carbon stocks (Cland) are the "capital" that produces carbon fluxes (GPP, NPP) through biological processes.

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

## Processing Steps

### Step 1: Pre-industrial Parameter Fitting ✅ COMPLETED
Fit the parameters of a Solow-Swan growth model to the results of a pre-industrial climate model simulation, assuming that the system does not respond at all to climate change.

**Parameters fitted:**
- Ksoil_0 (specified a priori)
- Kresp_0 (plant respiration fraction)
- Ktfp_0 (total factor productivity)
- alpha (production function exponent)

**Climate sensitivity parameters set to zero:**
- Ktfp_tas0, Ktfp_tas1, Ktfp_pr0, Ktfp_pr1 = 0

**Data used:** piControl simulation data

### Step 2: CO2 Fertilization Effect ✅ COMPLETED
Use the SSP585bgc simulation (where the biosphere sees the CO2 increase but the physics of the climate system does not) to tune a parameter, **Ktfp_co2**, which indicates the sensitivity of Ktfp to CO2 increase.

**Parameters fitted:**
- Ktfp_co2 (CO2 fertilization sensitivity)

**Parameters from Step 1 used as starting values:**
- Ksoil_0, Kresp_0, Ktfp_0, alpha

**CO2-dependent Ktfp equation:**
```
Ktfp = Ktfp_0 * tas_factor * pr_factor * co2_factor
```
Where:
- `co2_factor = (1 + Ktfp_co2) * ((co2/co2_0) / (Ktfp_co2 + co2/co2_0))`
- `co2_0 = 284.318604` ppm (pre-industrial reference concentration)

**Data used:** Concatenated historical + SSP585bgc simulation data + historical-SSP585 CO2 concentrations

### Step 3: Climate Sensitivity Estimation ✅ COMPLETED
Use the SSP585 simulation to estimate climate sensitivity parameters for temperature and precipitation effects on total factor productivity (Ktfp).

**Parameters fitted:**
- **Ktfp_tas0**: Reference temperature (°C) - mean temperature from piControl simulation
- **Ktfp_tas1**: Temperature sensitivity coefficient - fractional change in Ktfp per °C deviation from reference
- **Ktfp_pr0**: Reference precipitation (mm/day) - mean precipitation from piControl simulation  
- **Ktfp_pr1**: Precipitation sensitivity coefficient - fractional change in Ktfp per mm/day deviation from reference

**Parameters from Steps 1-2 used as starting values:**
- Ksoil_0, Kresp_0, Ktfp_0, alpha, Ktfp_co2

**Climate-dependent Ktfp equation:**
```
Ktfp = Ktfp_0 * tas_factor * pr_factor * co2_factor
```
Where:
- `tas_factor = 1 + Ktfp_tas1 * (tas - Ktfp_tas0)`
- `pr_factor = 1 + Ktfp_pr1 * (pr - Ktfp_pr0)`

**Data used:** Concatenated historical + SSP585 simulation data

**✅ OPTIMIZATION WORKING:** The optimization now successfully changes initial guess values and finds optimal solutions for climate sensitivity parameters.

### Step 4: Validation ✅ COMPLETED
Rerun Step 2's scenario (CO2 fertilization data) using all coefficients found in Step 3 (including climate sensitivities), without additional optimization.

**Purpose:** Validate the complete model by testing it on the CO2 fertilization dataset with all fitted parameters from Steps 1-3.

**No new parameters fitted:** All parameters from previous steps are used as fixed values.

**Data used:** Same as Step 2 (concatenated historical + SSP585bgc simulation data)

## New Climate Sensitivity Parameterization

The project now uses a sophisticated climate sensitivity parameterization that separates reference climate states from sensitivity coefficients:

### Temperature Sensitivity
- **Ktfp_tas0**: Reference temperature (°C) - typically the mean temperature from piControl simulation (~20.57°C)
- **Ktfp_tas1**: Temperature sensitivity coefficient - represents fractional change in Ktfp per °C deviation from reference

### Precipitation Sensitivity  
- **Ktfp_pr0**: Reference precipitation (mm/day) - typically the mean precipitation from piControl simulation (~3.26 mm/day)
- **Ktfp_pr1**: Precipitation sensitivity coefficient - represents fractional change in Ktfp per mm/day deviation from reference

### Physical Interpretation
This approach is more physically meaningful because:
- It separates the reference climate state from the sensitivity coefficients
- It allows proper linearization around a reference point
- The sensitivity coefficients are interpretable as fractional changes per unit deviation
- It avoids issues with simple linear scaling that can lead to negative or unrealistic values

### Parameter Bounds
- **Ktfp_tas0**: (10.0, 30.0) °C - reasonable temperature range
- **Ktfp_tas1**: (-0.99, 0.99) - fractional sensitivity bounds
- **Ktfp_pr0**: (1.0, 10.0) mm/day - reasonable precipitation range
- **Ktfp_pr1**: (-0.99, 0.99) - fractional sensitivity bounds

## Code Architecture

The project has been refactored into a modular structure with a clean parameter optimization approach:

- **`main.py`**: Command-line interface and orchestration
- **`step_utils.py`**: Shared utilities, BGC simulation, parameter optimization
- **`step1.py`**: Pre-industrial parameter fitting
- **`step2.py`**: CO2 fertilization effects
- **`step3.py`**: Climate sensitivity estimation
- **`step4.py`**: Validation step
- **`plotting_utils.py`**: PDF book generation

### New Clean Parameter Optimization Approach

The project now uses a clean, explicit parameter management system:

```python
def optimize_parameters(fixed_params, params_to_optimize, data_df, co2_df=None):
    """
    Optimize parameters for BGC simulation using a clean, explicit approach.
    
    Args:
        fixed_params (dict): Dictionary of parameter names and their fixed values
        params_to_optimize (list): List of parameter names to optimize
        data_df (pd.DataFrame): Data for fitting (must contain year, tas, pr, npp columns)
        co2_df (pd.DataFrame, optional): CO2 concentration data
    """
```

This approach makes parameter management much clearer and more maintainable.

## Features
- **Multi-step analysis**: Sequential parameter fitting with inheritance between steps
- **Multi-region processing**: Run simulations for all countries and models
- **Smart parameter optimization**: Only optimize parameters not provided by user
- **Batch processing**: Process multiple regions/models with single command
- **CO2 integration**: Historical and future CO2 concentration data support
- **Advanced climate sensitivity**: Sophisticated temperature and precipitation parameterization
- **Comprehensive output**: Timestamped CSV files with all fitted parameters
- **PDF visualization**: Automatic generation of PDF books for results visualization
- **Virtual environment**: Isolated Python environment for dependencies
- **Flexible parameter control**: Set specific parameters to zero while optimizing others
- **Clean parameter management**: Explicit fixed vs. optimized parameter handling

## Getting Started

### 1. Setup Virtual Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Individual Steps
```bash
# Step 1: Pre-industrial parameter fitting
python main.py --step step1 --region "Zimbabwe" --model "ACCESS-ESM1-5"

# Step 2: CO2 fertilization effects
python main.py --step step2 --region "Zimbabwe" --model "ACCESS-ESM1-5"

# Step 3: Climate sensitivity estimation
python main.py --step step3 --region "Zimbabwe" --model "ACCESS-ESM1-5"

# Step 4: Validation
python main.py --step step4 --region "Zimbabwe" --model "ACCESS-ESM1-5"
```

### 3. Run Complete Analysis
```bash
# Run all steps sequentially (including PDF generation)
python main.py --step all --region "Zimbabwe" --model "ACCESS-ESM1-5"

# Run for multiple regions
python main.py --step all --regions "Zimbabwe" "Zambia" --models "ACCESS-ESM1-5"
```

### 4. Run with Fixed Parameters
```bash
# Fix specific parameters for any step
python main.py --step step1 --Ksoil_0 0.05 --region "Zimbabwe"
python main.py --step step2 --Ktfp_co2 0.1 --region "Zimbabwe"

# Set some climate sensitivity parameters to zero while optimizing others
python main.py --step step3 --Ktfp_tas0 20.57 --Ktfp_pr0 3.26 --region "Zimbabwe"
```

### 5. Generate PDF Books
```bash
# Generate PDF books from existing results
python main.py --create-pdf-books
```

## Input Data
Place your input CSV files in `data/input/`. The project uses:
- `Data_regression_piControl.csv` - Pre-industrial control data (Step 1)
- `Data_regression_historical.csv` - Historical data (Steps 2-4)
- `Data_regression_ssp585bgc.csv` - SSP585 biogeochemical scenario data (Steps 2, 4)
- `Data_regression_ssp585.csv` - SSP585 scenario data (Step 3)
- `historical-ssp585_co2.csv` - Historical and future CO2 concentrations (Steps 2-4)

## Output Files
The model generates timestamped output files in `data/output/`:
- `fitted_parameters_all_step1_YYYYMMDD_HHMMSS.csv` - Step 1 fitted parameters
- `fitted_parameters_all_step2_YYYYMMDD_HHMMSS.csv` - Step 2 fitted parameters (including Ktfp_co2)
- `fitted_parameters_all_step3_YYYYMMDD_HHMMSS.csv` - Step 3 fitted parameters (including climate sensitivities)
- `fitted_parameters_all_step4_YYYYMMDD_HHMMSS.csv` - Step 4 validation results
- `simulation_results_{region}_{model}_step{N}_YYYYMMDD_HHMMSS.csv` - Individual simulation results

## PDF Books
The system automatically generates three PDF books when running all steps:
- **Book 1**: Step 1 results (GPP data vs. GPP model)
- **Book 2**: Step 2 results (GPP data vs. GPP model)  
- **Book 3**: Step 3 vs. Step 4 comparison (GPP data vs. GPP model for both steps)

## Current Status
- **Step 1**: ✅ **COMPLETED** - Pre-industrial parameter fitting
- **Step 2**: ✅ **COMPLETED** - CO2 fertilization effect estimation
- **Step 3**: ✅ **COMPLETED** - Climate sensitivity parameter estimation with new parameterization
- **Step 4**: ✅ **COMPLETED** - Validation step
- **Code Refactoring**: ✅ **COMPLETED** - Modular architecture with clean parameter optimization approach
- **Virtual Environment**: ✅ **COMPLETED** - Isolated Python environment with all dependencies
- **PDF Visualization**: ✅ **COMPLETED** - Automatic PDF book generation
- **Parameter Structure**: ✅ **COMPLETED** - Advanced climate sensitivity parameterization

## Command Line Options
- `--step`: Specify step to run ("step1", "step2", "step3", "step4", "all")
- `--region` / `--regions`: Specify single region or list of regions
- `--model` / `--models`: Specify single model or list of models
- `--Ksoil_0`, `--Kresp_0`, `--Ktfp_0`, `--alpha`: Fix specific parameters
- `--Ktfp_co2`: Fix CO2 fertilization parameter
- `--Ktfp_tas0`, `--Ktfp_tas1`, `--Ktfp_pr0`, `--Ktfp_pr1`: Set climate sensitivity parameters
- `--create-pdf-books`: Generate PDF books from existing results

## Dependencies
- pandas==2.0.3 - Data manipulation and analysis
- numpy==1.24.4 - Numerical operations
- scipy==1.10.1 - Scientific computing and optimization
- scikit-learn==1.3.2 - Machine learning utilities
- statsmodels==0.14.1 - Statistical modeling
- matplotlib==3.7.2 - Plotting and PDF generation

## License
MIT License 