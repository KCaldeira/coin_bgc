# COIN-BGC: JSON-Driven Flexible Workflow System

A flexible, JSON-driven implementation of the COIN-BGC (Carbon-Oxygen-Interactive-Network Biogeochemical Cycle) model for land-surface climate change simulation using a Solow-Swan growth model approach.

## Overview

This implementation provides a flexible, configuration-driven architecture for parameter optimization and forward simulation of land-surface biogeochemical cycles under climate change scenarios. The system uses JSON workflow files to define optimization sequences, making it easy to modify workflows without changing code.

## Architecture

### Core Components

1. **`main.py`**: Main controller with three-step execution (load data/config, execute workflow, generate output)
2. **`CoinBGC` Class**: The core model implementation with unified optimization capabilities
3. **`WorkflowConfigLoader`**: JSON workflow configuration loading and validation
4. **`workflow_config.py`**: Configuration management and parameter resolution system

### JSON Workflow System

The system uses JSON configuration files to define flexible optimization workflows:

- **Step Types**: `calculation`, `optimization` (handles both single and multi-dataset optimizations)
- **Parameter Sources**: `global` (user input), `step` (from previous step), `value` (direct value)
- **Data Sources**: Flexible assignment of datasets to optimization steps
- **Parameter Bounds**: Simple `[lower, initial, upper]` format for all optimization parameters

### Example JSON Configuration

```json
{
  "workflow_name": "COIN-BGC Standard Pipeline",
  "global_parameters": {
    "Ksoil_0": {"source": "user_input"},
    "alpha": {"source": "user_input"}
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
      "knowns": {
        "Ksoil_0": {"source": "global"},
        "Kresp_0": {"source": "step", "step": "step2_1"}
      },
      "unknowns": {
        "Ktfp_tas1": {"range": [-0.4, 0.001, 0.4]}
      }
    }
  ]
}
```

### Unified Optimization Architecture

- **Single Method**: `optimize_parameters()` handles both single and multiple datasets
- **Flexible Parameter Management**: Complete parameter sets (known + optimized) passed between steps  
- **Fail-Fast Validation**: Missing parameters or configurations cause immediate failure
- **Consistent Interface**: Same optimization method for all step types

## Usage

### Command Line Interface

```bash
# Run with default workflow
python main.py --alpha=0.5 --Ksoil_0=0.025 --regions "Zimbabwe" --models "ACCESS-ESM1-5"

# Run with custom workflow
python main.py --alpha=0.5 --Ksoil_0=0.025 --regions "Brazil,China" --models "ACCESS-ESM1-5" --json custom_workflow.json

# Run multiple regions and models
python main.py --alpha=0.7 --Ksoil_0=0.05 --regions "Brazil,China,Zimbabwe" --models "ACCESS-ESM1-5"
```

### Required Parameters

- `--alpha`: Production function exponent (required)
- `--Ksoil_0`: Soil respiration parameter - inverse time constant for heterotrophic respiration (required)
- `--regions`: Comma-separated list of regions to analyze (required)
- `--models`: Comma-separated list of climate models to analyze (required)
- `--json`: JSON workflow configuration file (optional, default: workflow_schema_example.json)

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

## Current Status: JSON-Driven Flexible System ✅

The system now features a **fully flexible JSON-driven workflow architecture**:

### Major Enhancements - JSON Workflow System

- **✅ JSON Configuration**: Complete workflow definition via JSON files
- **✅ Flexible Step Execution**: Dynamic optimization sequences based on JSON config
- **✅ Unified Optimization**: Single `optimize_parameters()` method handles all cases
- **✅ Parameter Flow Management**: Automatic parameter passing between workflow steps  
- **✅ Command Line Interface**: Full argument parsing with user-specified parameters
- **✅ Fail-Fast Validation**: Immediate failure when configurations are incomplete
- **✅ Multiple Workflow Support**: Easy to create custom workflows for different research questions

### Architecture Benefits

- **Complete Flexibility**: Define any optimization sequence in JSON without code changes
- **Parameter Reuse**: Results from previous steps automatically flow to subsequent steps  
- **Dataset Flexibility**: Use any combination of data sources per optimization step
- **Easy Modification**: Change workflows by editing JSON files, not code
- **Self-Documenting**: JSON workflows serve as documentation of analysis approach
- **Validation**: Built-in configuration validation catches errors early

## Key Features

- **Fail Fast Philosophy**: No error checking, immediate failure on missing information
- **Unified Architecture**: Single optimization method handles both single and multi-dataset cases
- **Clean Parameter Management**: Simple `[lower, initial, upper]` bounds format
- **Flexible Data Sources**: Support for any combination of datasets per step
- **Complete Parameter Flow**: Both known and optimized parameters passed between steps
- **Professional Output**: Timestamped directories with comprehensive results
- **Modular Design**: Clear separation between configuration, execution, and output

## Next Steps

The flexible JSON workflow system enables:

1. **Custom Workflows**: Create specialized JSON configurations for different research questions
2. **Production Runs**: Execute large-scale analyses with consistent, reproducible workflows
3. **Workflow Optimization**: Fine-tune optimization sequences based on performance analysis
4. **Alternative Approaches**: Easily test different parameter optimization strategies
5. **Collaborative Research**: Share and version control workflow definitions as JSON files
6. **Integration**: Extend the system with additional step types and data sources

## File Structure

```
coin_bgc/
├── main.py                           # Main controller (JSON-driven)
├── coin_bgc.py                      # Core model with flexible workflow execution
├── workflow_config.py               # JSON configuration loading and validation
├── workflow_schema_example.json     # Standard 6-step pipeline example
├── coin_bgc_econ.py                # Original production implementation (preserved)
├── main_econ.py                    # Original main function (preserved)
└── data/
    ├── input/                      # Input CSV files
    └── output/                     # Timestamped output directories
``` 