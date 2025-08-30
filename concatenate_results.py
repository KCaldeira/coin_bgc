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
    """Concatenate all CSV parameter files."""
    print("\n💾 Concatenating CSV parameter files...")
    
    # Find all unique substep CSV patterns
    csv_patterns = set()
    for output_dir in output_dirs:
        csv_files = glob.glob(os.path.join(output_dir, "substep_parameters_*.csv"))
        for csv_file in csv_files:
            # Extract pattern (everything before the timestamp)
            filename = os.path.basename(csv_file)
            # Pattern: substep_parameters_stepX_X_schema_timestamp.csv
            parts = filename.split('_')
            if len(parts) >= 5:
                pattern = '_'.join(parts[:-1]) + '_*.csv'  # Remove timestamp part
                csv_patterns.add(pattern)
    
    # Concatenate each CSV type
    for pattern in csv_patterns:
        print(f"   🔗 Processing pattern: {pattern}")
        all_dataframes = []
        
        for output_dir in output_dirs:
            matching_files = glob.glob(os.path.join(output_dir, pattern))
            for csv_file in matching_files:
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        df['source_run'] = os.path.basename(output_dir)
                        all_dataframes.append(df)
                except Exception as e:
                    print(f"     ⚠️  Failed to read {csv_file}: {e}")
        
        if all_dataframes:
            # Combine all dataframes
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Create output filename
            # Convert pattern back to specific filename
            pattern_base = pattern.replace('_*.csv', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{pattern_base}_merged_{timestamp}.csv"
            output_path = os.path.join(merged_dir, output_filename)
            
            # Save merged file
            merged_df.to_csv(output_path, index=False)
            print(f"     ✅ Merged {len(all_dataframes)} files → {output_filename} ({len(merged_df)} rows)")
        else:
            print(f"     ⚠️  No data files found for pattern {pattern}")

def concatenate_execution_reports(output_dirs, merged_dir):
    """Concatenate workflow execution reports."""
    print("\n📊 Concatenating execution reports...")
    
    all_reports = []
    for output_dir in output_dirs:
        report_files = glob.glob(os.path.join(output_dir, "workflow_execution_report_*.csv"))
        for report_file in report_files:
            try:
                df = pd.read_csv(report_file)
                if not df.empty:
                    df['source_run'] = os.path.basename(output_dir)
                    all_reports.append(df)
            except Exception as e:
                print(f"   ⚠️  Failed to read {report_file}: {e}")
    
    if all_reports:
        merged_reports = pd.concat(all_reports, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(merged_dir, f"workflow_execution_report_merged_{timestamp}.csv")
        merged_reports.to_csv(output_path, index=False)
        print(f"   ✅ Merged execution reports → workflow_execution_report_merged_{timestamp}.csv ({len(merged_reports)} rows)")
        
        # Print summary statistics
        total_combinations = len(merged_reports)
        successful = len(merged_reports[merged_reports['status'] == 'SUCCESS'])
        failed = len(merged_reports[merged_reports['status'] == 'FAILED'])
        success_rate = (successful / total_combinations * 100) if total_combinations > 0 else 0
        
        print(f"   📈 Overall Success Rate: {success_rate:.1f}% ({successful}/{total_combinations})")
        if failed > 0:
            print(f"   ❌ Failed Combinations: {failed}")
            # Show error summary
            error_counts = merged_reports[merged_reports['status'] == 'FAILED']['error_type'].value_counts()
            for error_type, count in error_counts.items():
                print(f"     • {error_type}: {count}")

def merge_pdf_books(output_dirs, merged_dir):
    """Create a summary of PDF books (individual PDFs can't be easily merged)."""
    print("\n📖 Cataloging PDF books...")
    
    all_pdfs = []
    for output_dir in output_dirs:
        pdf_files = glob.glob(os.path.join(output_dir, "*.pdf"))
        for pdf_file in pdf_files:
            all_pdfs.append({
                'source_run': os.path.basename(output_dir),
                'pdf_filename': os.path.basename(pdf_file),
                'pdf_path': pdf_file,
                'file_size_mb': os.path.getsize(pdf_file) / (1024*1024)
            })
    
    if all_pdfs:
        pdf_catalog = pd.DataFrame(all_pdfs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(merged_dir, f"pdf_catalog_merged_{timestamp}.csv")
        pdf_catalog.to_csv(output_path, index=False)
        print(f"   ✅ Created PDF catalog → pdf_catalog_merged_{timestamp}.csv ({len(all_pdfs)} PDFs)")
        print(f"   📚 Total PDF size: {pdf_catalog['file_size_mb'].sum():.1f} MB")
        
        # Copy PDFs to merged directory with prefixes to avoid conflicts
        pdf_merged_dir = os.path.join(merged_dir, "pdfs")
        os.makedirs(pdf_merged_dir, exist_ok=True)
        
        for _, row in pdf_catalog.iterrows():
            source_path = row['pdf_path']
            # Prefix with run name to avoid conflicts
            new_filename = f"{row['source_run']}_{row['pdf_filename']}"
            dest_path = os.path.join(pdf_merged_dir, new_filename)
            
            try:
                import shutil
                shutil.copy2(source_path, dest_path)
            except Exception as e:
                print(f"     ⚠️  Failed to copy {row['pdf_filename']}: {e}")
        
        print(f"   📁 Copied all PDFs to: {pdf_merged_dir}")

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
    
    # Perform concatenation
    concatenate_csv_files(output_dirs, merged_dir)
    concatenate_execution_reports(output_dirs, merged_dir)
    merge_pdf_books(output_dirs, merged_dir)
    
    print(f"\n✅ Concatenation completed successfully!")
    print(f"📁 Merged results saved to: {merged_dir}")

if __name__ == "__main__":
    main()