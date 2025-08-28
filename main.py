#!/usr/bin/env python3
"""
COIN-BGC: Main Controller for JSON-Driven Flexible Workflow System

This is the main entry point for the new flexible JSON-driven workflow system.
It orchestrates the complete analysis pipeline with three main phases:

Step 1: Read in data, JSON configuration files, etc.
Step 2: Perform processing steps (optimization workflows)  
Step 3: Perform final simulations and produce output

Usage:
    python main.py [workflow_config.json]

If no workflow file is specified, uses workflow_schema_example.json
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Import our configuration system
from workflow_config import WorkflowConfigLoader, WorkflowExecutor

# Import the flexible workflow implementation (to be modified)
from coin_bgc import CoinBGC


class CoinBGCController:
    """Main controller for JSON-driven COIN-BGC workflow execution."""
    
    def __init__(self, workflow_file: str, alpha: float, Ksoil_0: float, regions: list, models: list, verbose: bool = False):
        """
        Initialize the controller with a workflow configuration file and parameters.
        
        Args:
            workflow_file: Path to JSON workflow configuration file
            alpha: Production function exponent
            Ksoil_0: Inverse time constant for soil respiration  
            regions: List of regions to process
            models: List of models to process
            verbose: Enable verbose output with detailed parameter tracking
        """
        self.workflow_file = workflow_file
        self.alpha = alpha
        self.Ksoil_0 = Ksoil_0
        self.regions = regions
        self.models = models
        self.verbose = verbose
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"data/output/run_{self.timestamp}"
        
        # Initialize components
        self.config_loader = WorkflowConfigLoader()
        self.workflow_config = None
        self.workflow_executor = None
        self.coin_bgc = None
        
        # Data containers
        self.datasets = {}
        self.results = {}
        
    def run(self):
        """Execute the complete three-step workflow."""
        print(f"=== COIN-BGC Flexible Workflow System ===")
        print(f"Workflow file: {self.workflow_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        try:
            # Step 1: Read in data, JSON files, etc.
            self.step1_load_data_and_config()
            
            # Step 2: Perform processing steps
            self.step2_execute_workflow()
            
            # Step 3: Perform final simulations and produce output
            self.step3_generate_final_output()
            
            print(f"✅ Workflow completed successfully!")
            print(f"📁 Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Workflow failed: {str(e)}")
            raise
    
    def step1_load_data_and_config(self):
        """
        Step 1: Read in data, JSON configuration files, etc.
        
        This step:
        - Loads and validates the JSON workflow configuration
        - Loads all required datasets (CSV files)
        - Initializes the workflow executor
        - Creates output directories
        - Validates that all required data sources are available
        """
        print("🔄 Step 1: Loading data and configuration...")
        
        # Load and validate workflow configuration
        print(f"  📋 Loading workflow configuration from {self.workflow_file}")
        self.workflow_config = self.config_loader.load_config(self.workflow_file)
        print(f"  ✅ Loaded workflow: {self.workflow_config.workflow_name}")
        
        # Initialize workflow executor
        self.workflow_executor = WorkflowExecutor()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"  📁 Created output directory: {self.output_dir}")
        
        # Initialize CoinBGC instance
        self.coin_bgc = CoinBGC()
        
        # Load all required datasets
        self._load_datasets()
        
        print("  ✅ Step 1 completed: Data and configuration loaded")
        print()
    
    def step2_execute_workflow(self):
        """
        Step 2: Perform processing steps (optimization workflows).
        
        This step:
        - Executes each workflow step in sequence
        - Handles calculation steps, optimization steps, and multi-optimization steps
        - Manages parameter flow between steps
        - Saves intermediate results for each step
        """
        print("🔄 Step 2: Executing workflow steps...")
        
        # Use provided regions and models
        regions = self.regions
        models = self.models
        
        print(f"  🌍 Processing {len(regions)} regions × {len(models)} models")
        print(f"  📊 Executing {len(self.workflow_config.steps)} workflow steps")
        print(f"  🎯 Regions: {', '.join(regions)}")
        print(f"  🎯 Models: {', '.join(models)}")
        
        # Execute workflow for each region/model combination
        for region in regions:
            for model in models:
                print(f"    🎯 Processing: {region} - {model}")
                self._execute_region_model_workflow(region, model)
        
        print("  ✅ Step 2 completed: All workflow steps executed")
        print()
    
    def step3_generate_final_output(self):
        """
        Step 3: Perform final simulations and produce output.
        
        This step:
        - Runs final simulations using optimized parameters
        - Generates CSV output files for all datasets and steps
        - Creates PDF visualization books
        - Saves final parameter files
        - Produces summary reports
        """
        print("🔄 Step 3: Generating final output...")
        
        # Run final simulations if specified in workflow
        if hasattr(self.workflow_config, 'final_simulations'):
            self._run_final_simulations()
        
        # Generate comprehensive output files
        self._save_all_results()
        
        # Generate PDF visualization books
        self._generate_pdf_books()
        
        print("  ✅ Step 3 completed: Final output generated")
        print()
    
    def _load_datasets(self):
        """Load all required CSV datasets."""
        print("  📊 Loading datasets...")
        
        # Define standard dataset paths
        data_paths = {
            'piControl': 'data/input/Data_regression_piControl.csv',
            'historical': 'data/input/Data_regression_historical.csv',
            'full': 'data/input/Data_regression_ssp585.csv',
            'bgc': 'data/input/Data_regression_ssp585-bgc.csv',
            '1pctCO2': 'data/input/Data_regression_1pctCO2.csv',
            '1pctCO2_bgc': 'data/input/Data_regression_1pctCO2-bgc.csv',
            'co2_data': 'data/input/historical-ssp585_co2.csv'
        }
        
        # Load each dataset that exists
        for name, path in data_paths.items():
            if os.path.exists(path):
                print(f"    Loading {name} from {path}")
                self.datasets[name] = self.coin_bgc.load_data(path)
            else:
                print(f"    ⚠️  Dataset {name} not found at {path}")
        
        print(f"  ✅ Loaded {len(self.datasets)} datasets")
    
    def _get_regions_and_models(self):
        """Extract unique regions and models from datasets."""
        regions = set()
        models = set()
        
        for dataset in self.datasets.values():
            if hasattr(dataset, 'region'):
                regions.update(dataset['region'].unique())
            if hasattr(dataset, 'model'):
                models.update(dataset['model'].unique())
        
        return sorted(regions), sorted(models)
    
    def _execute_region_model_workflow(self, region, model):
        """Execute the complete workflow for a specific region/model combination."""
        # This will be implemented to call the flexible workflow execution
        # in the modified coin_bgc.py
        
        # For now, placeholder that shows the structure
        step_results = {}
        
        for step in self.workflow_config.steps:
            print(f"      🔧 Executing step: {step.name} ({step.step_type})")
            
            # Execute the workflow step using the new flexible method
            global_params = {'alpha': self.alpha, 'Ksoil_0': self.Ksoil_0}
            
            if self.verbose:
                self._print_parameter_universe_before_step(step, step_results, global_params)
            
            step_result = self.coin_bgc.execute_workflow_step(
                step, region, model, self.datasets, step_results, global_params, self.verbose
            )
            step_results[step.name] = step_result
            
            if self.verbose:
                self._print_parameter_universe_after_step(step, step_result)
        
        # Store results for this region/model
        if region not in self.results:
            self.results[region] = {}
        self.results[region][model] = step_results
    
    def _print_parameter_universe_before_step(self, step, step_results, global_params):
        """Print the complete parameter universe before executing a step."""
        print(f"        📊 PARAMETER UNIVERSE BEFORE {step.name.upper()}:")
        
        # Global parameters (user-specified)
        print(f"        🌍 Global Parameters (user-specified):")
        for param_name, value in global_params.items():
            print(f"          {param_name}: {value:.6f}")
        
        # Parameters from previous steps
        if step_results:
            print(f"        📈 Results from Previous Steps:")
            for step_name, results in step_results.items():
                print(f"          From {step_name}:")
                for param_name, value in results.items():
                    print(f"            {param_name}: {value:.6f}")
        
        # Parameters that will be known for this step
        print(f"        🔒 Known Parameters for {step.name}:")
        for param_name, param_spec in step.knowns.items():
            print(f"          {param_name}: source={param_spec.source}")
        
        # Parameters that will be optimized
        if hasattr(step, 'unknowns') and step.unknowns:
            print(f"        🎯 Parameters to Optimize in {step.name}:")
            for param_name, param_spec in step.unknowns.items():
                bounds = getattr(param_spec, 'range', 'unknown')
                print(f"          {param_name}: bounds={bounds}")
        
        print()
    
    def _print_parameter_universe_after_step(self, step, step_result):
        """Print the complete parameter universe after executing a step."""
        print(f"        📊 PARAMETER UNIVERSE AFTER {step.name.upper()}:")
        print(f"        ✅ Results from {step.name}:")
        for param_name, value in step_result.items():
            print(f"          {param_name}: {value:.6f}")
        print()
    
    def _run_final_simulations(self):
        """Run final simulations using optimized parameters."""
        print("  🎯 Running final simulations...")
        # This will be implemented to run final simulations
        # using the optimized parameters from all steps
        pass
    
    def _save_all_results(self):
        """Save all results to CSV files."""
        print("  💾 Saving results to CSV files...")
        # This will be implemented to save all parameter files
        # and simulation results to the output directory
        pass
    
    def _generate_pdf_books(self):
        """Generate PDF visualization books."""
        print("  📖 Generating PDF visualization books...")
        # This will be implemented to create the PDF books
        # showing all the optimization results and charts
        pass


def main():
    """Main entry point for the COIN-BGC flexible workflow system."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="COIN-BGC Flexible Workflow System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Use default workflow
  python main.py workflow_schema_example.json # Use specific workflow
  python main.py custom_workflow.json        # Use custom workflow
        """
    )
    
    parser.add_argument(
        '--json', 
        default='workflow_schema_example.json',
        help='JSON workflow configuration file (default: workflow_schema_example.json)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        required=True,
        help='Production function exponent (required)'
    )
    
    parser.add_argument(
        '--Ksoil_0',
        type=float,
        required=True,
        help='Inverse time constant for soil respiration (required)'
    )
    
    parser.add_argument(
        '--regions',
        type=str,
        required=True,
        help='Regions to process (comma-separated or single region)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        required=True,
        help='Models to process (comma-separated or single model)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output showing detailed parameter tracking'
    )
    
    args = parser.parse_args()
    
    # Validate workflow file exists
    if not os.path.exists(args.json):
        print(f"❌ Error: Workflow file '{args.json}' not found")
        sys.exit(1)
    
    # Parse regions and models
    regions = [r.strip() for r in args.regions.split(',')]
    models = [m.strip() for m in args.models.split(',')]
    
    # Create and run the controller
    controller = CoinBGCController(args.json, args.alpha, args.Ksoil_0, regions, models, args.verbose)
    controller.run()


if __name__ == "__main__":
    main()