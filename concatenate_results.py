#!/usr/bin/env python3
"""
Concatenation Script for COIN-BGC Parallel Runs

This script merges results from multiple parallel runs into unified outputs.
It handles:
- CSV parameter files
- PDF visualization books  
- Execution reports
- Timing reports

Usage:
    python concatenate_results.py [output_pattern]
    
Example:
    python concatenate_results.py "run_20250830_*"
"""

import os
import sys
import glob
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

def find_output_directories(pattern=None):
    """Find all output directories matching the pattern."""
    if pattern is None:
        pattern = "run_*"
    
    output_dirs = glob.glob(f"data/output/{pattern}")
    output_dirs = [d for d in output_dirs if os.path.isdir(d)]
    
    if not output_dirs:
        print(f"❌ No output directories found matching pattern: {pattern}")
        return []
    
    print(f"📁 Found {len(output_dirs)} output directories:")
    for d in sorted(output_dirs):
        print(f"   • {os.path.basename(d)}")
    
    return sorted(output_dirs)

def concatenate_csv_files(output_dirs, merged_dir):
    """Concatenate all standard CSV output files."""
    print("\n💾 Concatenating CSV files...")
    
    # Standard file types to concatenate
    file_types = [
        "substep_parameters_*.csv",
        "timing_report_*.csv", 
        "substep_timing_matrix_*.csv",
        "substep_timing_summary_*.csv",
        "workflow_execution_report_*.csv"
    ]
    
    for file_pattern in file_types:
        print(f"   🔗 Processing: {file_pattern}")
        all_files = []
        all_dataframes = []
        
        # Collect all matching files from all directories
        for output_dir in output_dirs:
            matching_files = glob.glob(os.path.join(output_dir, file_pattern))
            all_files.extend(matching_files)
        
        if not all_files:
            print(f"     ⚠️  No files found matching {file_pattern}")
            continue
            
        # Read and combine all matching files
        for csv_file in all_files:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Add source run information
                    df['source_run'] = os.path.basename(os.path.dirname(csv_file))
                    all_dataframes.append(df)
            except Exception as e:
                print(f"     ⚠️  Failed to read {csv_file}: {e}")
        
        if all_dataframes:
            # Combine all dataframes
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = file_pattern.replace('_*.csv', '').replace('*.csv', 'files')
            output_filename = f"{base_name}_merged_{timestamp}.csv"
            output_path = os.path.join(merged_dir, output_filename)
            
            # Save merged file
            merged_df.to_csv(output_path, index=False)
            print(f"     ✅ Merged {len(all_dataframes)} files → {output_filename} ({len(merged_df)} rows)")
        else:
            print(f"     ⚠️  No valid data found for {file_pattern}")

def merge_pdf_books(output_dirs, merged_dir):
    """Merge PDF books by step/datatype combination across all parallel runs."""
    print("\n📖 Merging PDF books by step/datatype...")
    
    try:
        from PyPDF2 import PdfMerger
        pdf_merger_available = True
    except ImportError:
        print("   ⚠️  PyPDF2 not available - will copy PDFs individually instead")
        pdf_merger_available = False
    
    # Collect all PDF files and group by step/datatype
    pdf_groups = {}  # {step_datatype: [list_of_pdf_paths]}
    
    for output_dir in output_dirs:
        pdf_files = glob.glob(os.path.join(output_dir, "*.pdf"))
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            # Parse filename: step2_6_BGC_full_jobs_YYYYMMDD_HHMMSS.pdf
            # Look for patterns like _Results_ or _jobs_ or _full_ to split on
            if "_Results_" in filename:
                # Extract step_datatype part (everything before _Results_)
                step_datatype = filename.split("_Results_")[0]
            elif "_jobs_" in filename:
                # Extract step_datatype part (everything before _jobs_)  
                step_datatype = filename.split("_jobs_")[0]
            elif filename.endswith('.pdf') and filename.count('_') >= 3:
                # Try to parse by removing timestamp at end
                # Format: step2_6_BGC_full_jobs_YYYYMMDD_HHMMSS.pdf
                parts = filename.replace('.pdf', '').split('_')
                if len(parts) >= 4:
                    # Take first 3-4 parts as step_datatype (step2_6_BGC or step2_6_BGC_full)
                    step_datatype = '_'.join(parts[:3])  # step2_6_BGC
            else:
                continue  # Skip files that don't match expected patterns
                
            if step_datatype not in pdf_groups:
                pdf_groups[step_datatype] = []
            pdf_groups[step_datatype].append(pdf_file)
    
    if not pdf_groups:
        print("   ℹ️  No PDF files found to merge")
        return
    
    pdfs_dir = os.path.join(merged_dir, "merged_pdfs")
    os.makedirs(pdfs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_count = 0
    
    # Merge PDFs for each step/datatype combination
    for step_datatype, pdf_files in pdf_groups.items():
        output_filename = f"{step_datatype}_merged_{timestamp}.pdf"
        output_path = os.path.join(pdfs_dir, output_filename)
        
        if pdf_merger_available and len(pdf_files) > 1:
            # Merge multiple PDFs into one
            try:
                merger = PdfMerger()
                for pdf_file in sorted(pdf_files):  # Sort for consistent order
                    merger.append(pdf_file)
                
                merger.write(output_path)
                merger.close()
                merged_count += 1
                print(f"   ✅ Merged {len(pdf_files)} PDFs → {output_filename}")
                
            except Exception as e:
                print(f"   ⚠️  Failed to merge {step_datatype}: {e}")
                # Fallback: copy individual files
                for i, pdf_file in enumerate(pdf_files):
                    fallback_name = f"{step_datatype}_job{i+1}_{timestamp}.pdf"
                    fallback_path = os.path.join(pdfs_dir, fallback_name)
                    import shutil
                    shutil.copy2(pdf_file, fallback_path)
        
        else:
            # Single PDF or no merger available - just copy
            if len(pdf_files) == 1:
                import shutil
                shutil.copy2(pdf_files[0], output_path)
                merged_count += 1
                print(f"   ✅ Copied single PDF → {output_filename}")
            else:
                # Multiple PDFs but no merger - copy individually with job numbers
                for i, pdf_file in enumerate(pdf_files):
                    copy_name = f"{step_datatype}_job{i+1}_{timestamp}.pdf"
                    copy_path = os.path.join(pdfs_dir, copy_name)
                    import shutil
                    shutil.copy2(pdf_file, copy_path)
                merged_count += len(pdf_files)
                print(f"   ✅ Copied {len(pdf_files)} individual PDFs for {step_datatype}")
    
    print(f"   📁 Created {merged_count} merged PDF files in: merged_pdfs/")

def copy_simulation_files(output_dirs, merged_dir):
    """Copy simulation CSV files with run prefixes."""
    print("\n📄 Copying simulation CSV files...")
    
    simulations_dir = os.path.join(merged_dir, "all_simulations") 
    os.makedirs(simulations_dir, exist_ok=True)
    
    sim_count = 0
    for output_dir in output_dirs:
        run_name = os.path.basename(output_dir)
        sim_files = glob.glob(os.path.join(output_dir, "simulation_*.csv"))
        for sim_file in sim_files:
            filename = os.path.basename(sim_file)
            # Prefix with run name to avoid conflicts  
            new_name = f"{run_name}_{filename}"
            dest_path = os.path.join(simulations_dir, new_name)
            try:
                import shutil
                shutil.copy2(sim_file, dest_path)
                sim_count += 1
            except Exception as e:
                print(f"     ⚠️  Failed to copy {filename}: {e}")
    
    print(f"   ✅ Copied {sim_count} simulation files to: all_simulations/")

def print_summary_statistics(merged_dir):
    """Print summary statistics from merged execution reports."""
    print("\n📊 Summary Statistics:")
    
    # Look for merged execution report
    exec_files = glob.glob(os.path.join(merged_dir, "workflow_execution_report_merged_*.csv"))
    if exec_files:
        try:
            df = pd.read_csv(exec_files[0])  # Use the most recent one
            
            total_combinations = len(df)
            if 'status' in df.columns:
                successful = len(df[df['status'] == 'SUCCESS'])
                failed = len(df[df['status'] == 'FAILED'])
                success_rate = (successful / total_combinations * 100) if total_combinations > 0 else 0
                
                print(f"   📈 Overall Success Rate: {success_rate:.1f}% ({successful}/{total_combinations})")
                if failed > 0:
                    print(f"   ❌ Failed Combinations: {failed}")
                    # Show error summary if error_type column exists
                    if 'error_type' in df.columns:
                        error_counts = df[df['status'] == 'FAILED']['error_type'].value_counts()
                        for error_type, count in error_counts.items():
                            print(f"     • {error_type}: {count}")
            else:
                print(f"   📋 Total combinations processed: {total_combinations}")
                
        except Exception as e:
            print(f"   ⚠️  Could not read execution report: {e}")
    else:
        print("   ℹ️  No execution report found")

def main():
    """Main concatenation function."""
    parser = argparse.ArgumentParser(description='Concatenate COIN-BGC parallel run results')
    parser.add_argument('pattern', nargs='?', default='run_*',
                       help='Pattern to match output directories (default: run_*)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for merged results (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("=== COIN-BGC Results Concatenation ===")
    print(f"Pattern: {args.pattern}")
    
    # Find output directories
    output_dirs = find_output_directories(args.pattern)
    if not output_dirs:
        sys.exit(1)
    
    # Create merged output directory
    if args.output_dir:
        merged_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_dir = f"data/output/merged_{timestamp}"
    
    os.makedirs(merged_dir, exist_ok=True)
    print(f"\n📁 Merged results will be saved to: {merged_dir}")
    
    # Perform concatenation and copying
    concatenate_csv_files(output_dirs, merged_dir)
    merge_pdf_books(output_dirs, merged_dir)
    copy_simulation_files(output_dirs, merged_dir)
    print_summary_statistics(merged_dir)
    
    print(f"\n✅ Concatenation completed successfully!")
    print(f"📁 Merged results saved to: {merged_dir}")
    print(f"📁 Merged PDFs by step/datatype in: {merged_dir}/merged_pdfs/")
    print(f"📁 Individual simulations in: {merged_dir}/all_simulations/")

if __name__ == "__main__":
    main()