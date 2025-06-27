# Coin BGC: Simple Biogeochemical Model Simulation

This project simulates biogeochemical carbon cycling for different regions (countries) using a simple model. The model is designed for educational and exploratory purposes, and can be extended to include climate change effects and parameter optimization.

## Features
- Reads historical biogeochemical data for multiple regions and models
- Filters data for a specific region and model
- Simulates carbon land pool (Cland) dynamics using a simple annual time-stepping model
- Easily extendable for more complex scenarios

## Model Equations
- **GPP = Ktfp * Cland^alpha**
- **NPP = (1 - Kresp) * GPP**
- **SOILresp = Ksoil * Cland**
- **dCland/dt = NPP - SOILresp**
- **Cland(t+1) = Cland(t) + dCland/dt**

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
Place your input CSV files in `data/input/`. The main script is set up to use `Data_regression_historical.csv` by default.

## Customization
- Edit `main.py` to change the region, model, or simulation parameters.
- Extend the model logic in `run_bgc_simulation` as needed.

## License
MIT License (add your preferred license here) 