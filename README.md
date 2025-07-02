# Coin BGC: Simple Biogeochemical Model Simulation

This project simulates biogeochemical carbon cycling for different regions (countries) using a simple model. The model is designed for educational and exploratory purposes, and can be extended to include climate change effects and parameter optimization.

## Features
- Reads historical biogeochemical data for multiple regions and models
- Filters data for a specific region and model
- Simulates carbon land pool (Cland) dynamics using a simple annual time-stepping model
- **Parameter optimization** using scipy.optimize to minimize prediction errors
- **Uncertainty estimation** with standard errors for all fitted parameters
- **Step 1 completed**: Base parameter estimation using piControl data
- Climate change impact analysis using different scenarios (Steps 2-5 planned)
- Easily extendable for more complex scenarios

## Model Equations
- **GPP = Ktfp * Cland^alpha**
- **NPP = (1 - Kresp) * GPP**
- **SOILresp = Ksoil * Cland**
- **dCland/dt = NPP - SOILresp**
- **Cland(t+1) = Cland(t) + dCland/dt**

Where Ksoil, Kresp, and Ktfp can be functions of temperature (tas) and precipitation (pr).

## Processing Sequence
The model follows a 5-step parameter estimation and validation process:

1. **✅ Base Parameter Estimation (piControl)**: With piControl data, assuming all parameters are scalar constants, estimate model parameters ignoring climate change effects. **COMPLETED**

2. **Total Factor Productivity (TFP) Estimation**: Use historical-bgc and ssp585-bgc data to determine total factor productivity changes over time.

3. **Climate Effect on GPP**: Use historical data to estimate climate sensitivity effects on GPP.

4. **SSP585 Prediction**: Use the previously computed total factor productivity and climate sensitivity to predict ssp585 results.

5. **Validation**: Compare ssp585 predictions to actual ssp585 results to validate the model.

## Getting Started
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python main.py
   ```

## Input Data
Place your input CSV files in `data/input/`. The project uses:
- `Data_regression_piControl.csv` - Pre-industrial control data (Step 1)
- `Data_regression_historical.csv` - Historical climate and biogeochemical data
- `Data_regression_ssp585.csv` - SSP585 scenario data
- `Data_regression_ssp585-bgc.csv` - SSP585 biogeochemical scenario data

## Output Files
The model generates several output files in `data/output/`:
- `{region}_{model}_step1_results.csv` - Detailed simulation results for each region/model
- `step1_fitted_parameters.csv` - Summary of all fitted parameters with uncertainties

## Current Status
- **Step 1**: ✅ Complete - Parameter optimization for Ksoil_0, Kresp_0, Ktfp_0, and alpha
- **Steps 2-5**: 🔄 Planned - TFP estimation, climate sensitivity, prediction, and validation

## Customization
- Edit `main.py` to change the region, model, or simulation parameters
- Extend the model logic in `run_bgc_simulation` as needed
- Modify the parameter estimation process in `step1_picontrol_parameter_estimation`

## License
MIT License (add your preferred license here) 