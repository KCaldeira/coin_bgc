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

Where Ksoil, Kresp, and Ktfp can be functions of temperature (tas), precipitation (pr), and CO2 concentration.

## Processing Steps

### Step 1: Pre-industrial Parameter Fitting ✅
Fit the parameters of a Solow-Swan growth model to the results of a pre-industrial climate model simulation, assuming that the system does not respond at all to climate change.

**Parameters to fit:**
- Ksoil_0 (specified a priori)
- Kresp_0 (plant respiration fraction)
- Ktfp_0 (total factor productivity)
- alpha (production function exponent)

**Climate sensitivity parameters set to zero:**
- Ksoil_tas, Ksoil_pr, Kresp_tas, Kresp_pr, Ktfp_tas, Ktfp_pr = 0

**Variant:** We might want to try fitting some or all of the climate sensitivity parameters at this stage.

### Step 2: CO2 Fertilization Effect (Planned)
Use the SSP585bgc simulation (where the biosphere sees the CO2 increase but the physics of the climate system does not) to tune a parameter, **Ktfp_co2**, which indicates the sensitivity of Ktfp to CO2 increase.

**New equation for Ktfp:**
```
Ktfp = Ktfp_0 * (1 + Ktfp_co2) * ((co2/co2_0) / (Ktfp_co2 + co2/co2_0))
```

### Step 3: Climate Sensitivity Estimation (Planned)
Use the SSP585 simulation to estimate some or all of the following climate sensitivity factors:
- Ksoil_tas (temperature sensitivity of soil respiration)
- Ksoil_pr (precipitation sensitivity of soil respiration)
- Kresp_tas (temperature sensitivity of plant respiration)
- Kresp_pr (precipitation sensitivity of plant respiration)
- Ktfp_tas (temperature sensitivity of total factor productivity)
- Ktfp_pr (precipitation sensitivity of total factor productivity)

## Features
- **Multi-region processing**: Run simulations for all countries and models
- **Smart parameter optimization**: Only optimize parameters not provided by user
- **Batch processing**: Process multiple regions/models with single command
- **Uncertainty estimation**: Standard errors for all fitted parameters
- **Flexible parameter specification**: Mix user-provided and optimized parameters
- **Comprehensive output**: Single CSV file with all fitted parameters

## Getting Started
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run for all regions and models:
   ```bash
   python main.py
   ```

3. Run for specific regions:
   ```bash
   python main.py --regions "Zimbabwe" "Zambia" --models "ACCESS-ESM1-5"
   ```

4. Run with fixed parameters:
   ```bash
   python main.py --Ksoil_0 0.05 --regions "Zimbabwe" "Zambia"
   ```

## Input Data
Place your input CSV files in `data/input/`. The project uses:
- `Data_regression_piControl.csv` - Pre-industrial control data (Step 1)
- `Data_regression_ssp585bgc.csv` - SSP585 biogeochemical scenario data (Step 2)
- `Data_regression_ssp585.csv` - SSP585 scenario data (Step 3)

## Output Files
The model generates output files in `data/output/`:
- `fitted_parameters_all_step1_YYYYMMDD_HHMMSS.csv` - All fitted parameters in single file
- `simulation_results_{region}_{model}_step1_YYYYMMDD_HHMMSS.csv` - Individual simulation results

## Current Status
- **Step 1**: ✅ Complete - Pre-industrial parameter fitting with uncertainty estimation
- **Step 2**: 🔄 Planned - CO2 fertilization effect estimation
- **Step 3**: 🔄 Planned - Climate sensitivity parameter estimation

## Command Line Options
- `--region` / `--regions`: Specify single region or list of regions
- `--model` / `--models`: Specify single model or list of models
- `--Ksoil_0`, `--Kresp_0`, `--Ktfp_0`, `--alpha`: Fix specific parameters
- `--Ksoil_tas`, `--Ksoil_pr`, etc.: Set climate sensitivity parameters

## Dependencies
- pandas - Data manipulation and analysis
- numpy - Numerical operations
- scikit-learn - Machine learning utilities
- statsmodels - Statistical modeling
- scipy - Scientific computing and optimization

## License
MIT License 