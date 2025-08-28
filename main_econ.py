#!/usr/bin/env python3
"""
COIN-BGC: Clean Implementation Command Line Interface

This is the main entry point for the clean COIN-BGC implementation.
It provides a command-line interface to run the complete analysis workflow.
"""

import argparse
import sys
import pandas as pd
import traceback
from typing import List

from coin_bgc_econ import run_main_analysis, load_data_for_analysis, load_co2_data, get_run_output_directory


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="COIN-BGC: Clean Implementation - Solow-Swan Growth Model for Land-Surface Climate Change Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis for a single region/model
  python main.py --regions "Zimbabwe" --models "ACCESS-ESM1-5" --Ksoil_0 0.1 --alpha 0.5

  # Run analysis for multiple regions
  python main.py --regions "Zimbabwe" "China" "Brazil" --models "ACCESS-ESM1-5" --Ksoil_0 0.1 --alpha 0.5

  # Run analysis for multiple models
  python main.py --regions "Zimbabwe" --models "ACCESS-ESM1-5" "CanESM5" --Ksoil_0 0.1 --alpha 0.5

  # Run with different parameters
  python main.py --regions "Zimbabwe" --models "ACCESS-ESM1-5" --Ksoil_0 0.025 --alpha -0.5

  # List available regions and models
  python main.py --list-data
        """
    )
    
    # Required parameters
    parser.add_argument(
        '--Ksoil_0', 
        type=float, 
        required=True,
        help='Soil respiration parameter (inverse time constant for heterotrophic respiration)'
    )
    parser.add_argument(
        '--alpha', 
        type=float, 
        required=True,
        help='Production function exponent'
    )
    
    # Region and model selection
    parser.add_argument(
        '--regions', 
        nargs='+', 
        type=str,
        help='List of regions to analyze (e.g., "Zimbabwe" "China" "Brazil"). If not specified, all available regions will be used.'
    )
    parser.add_argument(
        '--models', 
        nargs='+', 
        type=str,
        help='List of climate models to analyze (e.g., "ACCESS-ESM1-5" "CanESM5"). If not specified, all available models will be used.'
    )
    
    # Data exploration
    parser.add_argument(
        '--list-data', 
        action='store_true',
        help='List available regions and models in the data files'
    )
    
    # Output options
    parser.add_argument(
        '--output-file', 
        type=str,
        help='Output file to save results (CSV format)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    # Optimization options
    parser.add_argument(
        '--max-iterations', 
        type=int, 
        default=100000,
        help='Maximum number of iterations for optimization (default: 100000)'
    )
    
    return parser.parse_args()


def list_available_data():
    """List available regions and models in the data files."""
    print("Loading data to find available regions and models...")
    
    try:
        # Load all data without filtering to see what's available
        piControl_data, full_data, bgc_data = load_data_for_analysis()
        
        print("\nAvailable Regions:")
        regions = sorted(piControl_data['region'].unique())
        for region in regions:
            print(f"  - {region}")
        
        print("\nAvailable Models:")
        models = sorted(piControl_data['model'].unique())
        for model in models:
            print(f"  - {model}")
        
        print(f"\nData Summary:")
        print(f"  piControl_data: {len(piControl_data)} rows")
        print(f"  full_data: {len(full_data)} rows")
        print(f"  bgc_data: {len(bgc_data)} rows")
        
        # Show sample of region/model combinations
        print(f"\nSample Region/Model Combinations:")
        combinations = piControl_data[['region', 'model']].drop_duplicates().head(10)
        for _, row in combinations.iterrows():
            print(f"  - {row['region']} / {row['model']}")
        
        if len(combinations) > 10:
            print(f"  ... and {len(piControl_data[['region', 'model']].drop_duplicates()) - 10} more combinations")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the data files are in the data/input/ directory.")
        sys.exit(1)


def get_all_available_regions_and_models():
    """Get all available regions and models from the data files."""
    try:
        # Load all data without filtering to see what's available
        piControl_data, full_data, bgc_data = load_data_for_analysis()
        
        regions = sorted(piControl_data['region'].unique())
        models = sorted(piControl_data['model'].unique())
        
        return regions, models
    except Exception as e:
        print(f"Error loading data to get available regions/models: {e}")
        print("Make sure the data files are in the data/input/ directory.")
        sys.exit(1)


def validate_parameters(args):
    """Validate command line parameters."""
    if args.Ksoil_0 <= 0:
        print("ERROR: Ksoil_0 must be positive")
        sys.exit(1)
    
    if args.alpha < 0 or args.alpha > 1:
        print("ERROR: alpha must be between 0 and 1")
        sys.exit(1)
    
    # Note: regions and models are now optional - if not specified, all available will be used


def save_results(results, output_file):
    """Save results to CSV file."""
    
    # Convert results dictionary to DataFrame
    rows = []
    for key, params in results.items():
        row = {'region_model': key}
        row.update(params)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Handle list-data option
    if args.list_data:
        list_available_data()
        return
    
    # Validate parameters
    validate_parameters(args)
    
    # Get regions and models (use all available if not specified)
    regions = args.regions
    models = args.models
    
    if not regions:
        print("No regions specified, using all available regions...")
        regions, _ = get_all_available_regions_and_models()
    
    if not models:
        print("No models specified, using all available models...")
        _, models = get_all_available_regions_and_models()
    
    # Print run information
    print("=== COIN-BGC Clean Implementation ===")
    print(f"Regions: {regions}")
    print(f"Models: {models}")
    print(f"Ksoil_0: {args.Ksoil_0}")
    print(f"alpha: {args.alpha}")
    if args.output_file:
        print(f"Output file: {args.output_file}")
    print("=" * 50)
    
    try:
        # Run the main analysis
        results = run_main_analysis(regions, models, args.Ksoil_0, args.alpha, args.max_iterations)
        
        # Print results summary
        print(f"\n=== Analysis Complete ===")
        print(f"Successfully processed {len(results)} region/model combinations")
        
        for key, params in results.items():
            print(f"\nResults for {key}:")
            for param, value in params.items():
                print(f"  {param}: {value:.6f}")
        
        # Save results if output file specified
        if args.output_file:
            save_results(results, args.output_file)
        
        print(f"\n=== Analysis Summary ===")
        print("All optimizations completed successfully!")
        print("The results contain the final optimized parameters for each region/model combination.")
        print("You can use these parameters for further analysis or visualization.")
        
        # Show output directory
        output_dir = get_run_output_directory()
        print(f"\nResults saved to: {output_dir}")
        print("Files created:")
        print(f"  - fitted_parameters_all_complete_*.csv (final parameters)")
        print(f"  - simulation_*_*.csv (model simulation results)")
        print(f"  - PDF books (if implemented)")
        
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
