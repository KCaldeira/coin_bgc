# Session Status: COIN-BGC Project - Solow-Swan Growth Model Approach

## Project Overview
This project simulates the behavior of the land-surface model under climate change using a Solow-Swan growth model of an economy. The model represents terrestrial carbon cycling as an economic system where carbon stocks (Cland) are the "capital" that produces carbon fluxes (GPP, NPP) through biological processes.

## Overall Goal
The overall goal is to simulate land-surface behavior under climate change using a Solow-Swan growth model. The system is under-determined by one parameter, so **Ksoil_0** (inverse time constant for heterotrophic respiration) is chosen a priori.

## Current State
- **Step 1 COMPLETED**: Pre-industrial parameter fitting using piControl data
- **Multi-region processing**: Can run for all countries and models simultaneously
- **Smart parameter optimization**: Only optimizes parameters not provided by user
- **Batch processing**: Single command processes multiple regions/models
- **Uncertainty estimation**: Standard errors for all fitted parameters
- **Comprehensive output**: Single CSV file with all fitted parameters

## Processing Strategy

### Step 1: Pre-industrial Parameter Fitting ✅ COMPLETED
**Goal**: Fit Solow-Swan growth model parameters to pre-industrial climate model simulation, assuming no climate change response.

**Parameters fitted:**
- Ksoil_0 (specified a priori)
- Kresp_0 (plant respiration fraction)
- Ktfp_0 (total factor productivity)
- alpha (production function exponent)

**Climate sensitivity parameters set to zero:**
- Ksoil_tas, Ksoil_pr, Kresp_tas, Kresp_pr, Ktfp_tas, Ktfp_pr = 0

**Variant consideration**: We might want to try fitting some or all climate sensitivity parameters at this stage.

### Step 2: CO2 Fertilization Effect 🔄 PLANNED
**Goal**: Use SSP585bgc simulation to tune Ktfp_co2 parameter for CO2 sensitivity.

**New Ktfp equation:**
```
Ktfp = Ktfp_0 * (1 + Ktfp_co2) * ((co2/co2_0) / (Ktfp_co2 + co2/co2_0))
```

**Data needed**: SSP585bgc simulation (biosphere sees CO2 increase, climate physics does not)

### Step 3: Climate Sensitivity Estimation 🔄 PLANNED
**Goal**: Use SSP585 simulation to estimate climate sensitivity parameters.

**Parameters to estimate:**
- Ksoil_tas (temperature sensitivity of soil respiration)
- Ksoil_pr (precipitation sensitivity of soil respiration)
- Kresp_tas (temperature sensitivity of plant respiration)
- Kresp_pr (precipitation sensitivity of plant respiration)
- Ktfp_tas (temperature sensitivity of total factor productivity)
- Ktfp_pr (precipitation sensitivity of total factor productivity)

**Data needed**: SSP585 simulation (full climate change scenario)

## Proposed Top-Level Program Structure

### Current Structure Analysis
The current `main.py` bottom section contains logic that:
1. Parses command line arguments
2. Determines which regions/models to run
3. Sets up user parameters
4. Loops through region/model combinations
5. Calls `run_single_region_model()` for each combination
6. Collects and saves results

### Proposed Modular Structure

#### 1. **Main Orchestrator Function**
```python
def run_complete_analysis(args, step="step1"):
    """
    Top-level function that orchestrates the complete analysis.
    
    Args:
        args: Parsed command line arguments
        step: Which step to run ("step1", "step2", "step3", or "all")
    """
```

#### 2. **Step-Specific Functions**
Each step would have its own dedicated function:

```python
def run_step1_analysis(args):
    """Step 1: Pre-industrial parameter fitting using piControl data"""
    
def run_step2_analysis(args):
    """Step 2: CO2 fertilization effect using SSP585bgc data"""
    
def run_step3_analysis(args):
    """Step 3: Climate sensitivity estimation using SSP585 data"""
```

#### 3. **Shared Infrastructure Functions**
Functions that are used across multiple steps:

```python
def setup_analysis_environment(args):
    """Setup output directories, validate data files, etc."""
    
def determine_regions_and_models_to_run(args):
    """Already exists - determines which combinations to process"""
    
def collect_and_save_results(all_results, step, args):
    """Collect results from all region/model combinations and save"""
```

### Implementation Strategy

#### Phase 1: Refactor Current Code
1. **Extract Step 1 logic**: Move the current main execution logic into `run_step1_analysis()`
2. **Create shared functions**: Extract common setup and result collection logic
3. **Create main orchestrator**: Build `run_complete_analysis()` that can call individual steps

#### Phase 2: Implement Step 2
1. **Create `run_step2_analysis()`**: Similar structure to Step 1 but with CO2-dependent Ktfp
2. **Add CO2 data handling**: Load and process SSP585bgc data with CO2 concentrations
3. **Modify Ktfp calculation**: Implement the new CO2-dependent equation
4. **Add Ktfp_co2 parameter**: Include in optimization and uncertainty estimation

#### Phase 3: Implement Step 3
1. **Create `run_step3_analysis()`**: Similar structure but with climate sensitivity parameters
2. **Add climate sensitivity optimization**: Include Ksoil_tas, Ksoil_pr, Kresp_tas, etc.
3. **Load SSP585 data**: Process full climate change scenario data
4. **Validate against observations**: Compare predictions with actual climate responses

### Command Line Interface Design

The main program would support:

```bash
# Run specific steps
python main.py --step step1
python main.py --step step2
python main.py --step step3

# Run all steps sequentially
python main.py --step all

# Run with specific parameters for any step
python main.py --step step1 --Ksoil_0 0.05 --regions "Zimbabwe" "Zambia"
python main.py --step step2 --Ktfp_co2 0.1 --regions "Zimbabwe"
```

### Data Flow Between Steps

1. **Step 1 → Step 2**: Step 1 results (Ksoil_0, Kresp_0, Ktfp_0, alpha) become inputs to Step 2
2. **Step 2 → Step 3**: Step 2 results (Ktfp_co2) plus Step 1 results become inputs to Step 3
3. **Final Output**: Comprehensive parameter set with all climate sensitivities

### Key Design Principles

1. **Modularity**: Each step is self-contained but can share infrastructure
2. **Flexibility**: Can run individual steps or the complete sequence
3. **Consistency**: Same optimization framework and uncertainty estimation across all steps
4. **Scalability**: Maintains multi-region processing capability
5. **Extensibility**: Easy to add new parameters or modify existing ones

### Questions for Discussion

1. **Data dependencies**: Should each step load its own data, or should we create a shared data loading function?

2. **Parameter inheritance**: How should parameters from previous steps be passed to subsequent steps? (e.g., should Step 2 use Step 1's fitted parameters as starting values?)

3. **Error handling**: How should we handle cases where one step fails for certain regions/models?

4. **Output organization**: Should each step create its own output files, or should we have a unified output structure?

5. **Validation**: How should we validate results between steps?

### Answers to Design Questions

1. **Data dependencies**: **Shared data loading function** - Create a common function that can load and validate data files for each step, ensuring consistency and reducing code duplication.

2. **Parameter inheritance**: **Parameters from each step should be passed to the next step** - Step 2 will use Step 1's fitted parameters (Ksoil_0, Kresp_0, Ktfp_0, alpha) as inputs, and Step 3 will use both Step 1 and Step 2 results as starting values.

3. **Error handling**: **Stop on failure for diagnosis** - At this stage, if there is a failure the code should stop so we can diagnose what is going wrong. This ensures we catch and fix issues early in development.

4. **Output organization**: **Each step creates its own output file with common timestamp** - Each step should create its own output file, but use a common timestamp so we can identify which case it is and so files from two runs will not overwrite each other.

5. **Validation**: **Worry about validation later** - Focus on implementing the core functionality first, then add validation and diagnostics as needed.

## Completed Today
- ✅ **Step 1 fully implemented**: Pre-industrial parameter fitting with optimization
- ✅ **Multi-region processing**: Can run for all 151 regions and 1 model
- ✅ **Smart parameter optimization**: Only optimizes unfixed parameters
- ✅ **Batch processing**: Single command processes multiple regions/models
- ✅ **Uncertainty estimation**: Standard errors for all fitted parameters
- ✅ **Comprehensive output**: Single timestamped CSV file with all parameters
- ✅ **Command line flexibility**: Options to specify regions, models, and parameters
- ✅ **Clean output**: Removed verbose print statements for batch processing
- ✅ **Documentation updated**: README.md and CONTINUATION.md reflect new approach
- ✅ **Phase 1 refactoring completed**: Modular structure implemented with main orchestrator and step-specific functions

## Technical Implementation
- **Model framework**: Solow-Swan growth model applied to carbon cycling
- **Parameter optimization**: L-BFGS-B algorithm with bounds
- **Uncertainty quantification**: Hessian-based standard errors
- **Data handling**: Efficient filtering and processing for multiple regions
- **Output management**: Timestamped files with comprehensive parameter summaries

## Next Steps for Implementation

### Immediate Priorities
1. **Implement Step 2 (CO2 fertilization)**
   - Add Ktfp_co2 parameter to model equations
   - Create step2_ssp585bgc_parameter_estimation function
   - Modify Ktfp calculation to include CO2 dependence
   - Test with SSP585bgc data

2. **Implement Step 3 (Climate sensitivity)**
   - Create step3_ssp585_parameter_estimation function
   - Add climate sensitivity parameters to optimization
   - Test with SSP585 data
   - Validate against observed climate responses

### Implementation Strategy
1. **Extend current framework**: Build on existing step1 implementation
2. **Maintain consistency**: Use same optimization and uncertainty estimation approach
3. **Preserve flexibility**: Keep ability to fix/optimize specific parameters
4. **Ensure scalability**: Maintain multi-region processing capability

### Data Requirements
- **Step 2**: SSP585bgc simulation data with CO2 concentrations
- **Step 3**: SSP585 simulation data with full climate change
- **Validation**: Comparison datasets for model validation

## Project Status Summary
- **Step 1**: ✅ **FULLY COMPLETED** - Pre-industrial parameter fitting with uncertainty estimation
- **Infrastructure**: ✅ **READY** - Multi-region processing, batch capabilities, clean output
- **Code Quality**: ✅ **EXCELLENT** - Well-documented, properly structured, scalable
- **Phase 1**: ✅ **COMPLETED** - Modular structure implemented with main orchestrator and step-specific functions
- **Next Phase**: 🔄 **READY TO BEGIN** - Steps 2-3 implementation with modular structure in place

## Key Advantages of Current Implementation
1. **Scalable**: Can process all regions/models efficiently
2. **Flexible**: Mix of user-provided and optimized parameters
3. **Robust**: Comprehensive error handling and uncertainty estimation
4. **Clean**: Minimal verbose output for batch processing
5. **Extensible**: Easy to add new parameters and steps

---
_Last updated: Step 1 fully implemented with multi-region processing, ready for Steps 2-3 implementation with proposed modular structure_ 