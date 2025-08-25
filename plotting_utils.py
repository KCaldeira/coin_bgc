"""
Plotting utilities for COIN-BGC simulation results.

This module creates PDF books showing simulation results for each step:
- Book 1: Step 1 results (GPP and NPP data vs model)
- Book 2: Step 2 results (GPP and NPP data vs model)  
- Book 3: Step 3 vs Step 4 comparison (4 lines per panel)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import os
from step_utils import get_run_output_directory

def create_step1_book(output_dir):
    """
    Create PDF book showing Step 1 results.
    
    Each panel shows:
    - GPP_data (heavy line)
 th    - GPP_model (fine line)
    """
    print("Creating Step 1 results book...")
    
    # Find Step 1 simulation results files
    step1_files = []
    if os.path.isdir(output_dir):
        step1_files = [f for f in os.listdir(output_dir) if f.startswith("simulation_results_") and "_step1" in f]
    
    if not step1_files:
        print("No Step 1 simulation results found.")
        return
    
    # Create PDF
    pdf_path = os.path.join(output_dir, "Step1_Results_Book.pdf")
    with PdfPages(pdf_path) as pdf:
        for file_path in sorted(step1_files):
            # Load simulation results
            full_path = os.path.join(output_dir, file_path)
            df = pd.read_csv(full_path)
            
            # Extract region and model from filename
            # Remove simulation_results_ prefix and step suffix
            clean_name = file_path.replace("simulation_results_", "")
            # Find the step pattern and remove everything from there
            for step in ["_step1", "_step2", "_step3", "_step4"]:
                if step in clean_name:
                    clean_name = clean_name.split(step)[0]
                    break
            parts = clean_name.split("_")
            region = parts[0]
            model = "_".join(parts[1:])  # Model name might contain underscores
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot GPP data only
            ax.plot(df['year'], df['gpp_data'], 'b-', linewidth=2, label='GPP Data', alpha=0.8)
            ax.plot(df['year'], df['GPP'], 'b-', linewidth=1, label='GPP Model', alpha=0.6)
            
            # Customize plot
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
            ax.set_title(f'Step 1 Results: {region} / {model}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add MSE information if available
            if 'final_mse' in df.columns:
                mse = df['final_mse'].iloc[0]
                ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"Step 1 book saved to: {pdf_path}")

def create_step2_book(output_dir):
    """
    Create PDF book showing Step 2 results.
    
    Each panel shows:
    - GPP_data (heavy line)
    - GPP_model (fine line)
    """
    print("Creating Step 2 results book...")
    
    # Find Step 2 simulation results files
    step2_files = []
    if os.path.isdir(output_dir):
        step2_files = [f for f in os.listdir(output_dir) if f.startswith("simulation_results_") and "_step2" in f]
    
    if not step2_files:
        print("No Step 2 simulation results found.")
        return
    
    # Create PDF
    pdf_path = os.path.join(output_dir, "Step2_Results_Book.pdf")
    with PdfPages(pdf_path) as pdf:
        for file_path in sorted(step2_files):
            # Load simulation results
            full_path = os.path.join(output_dir, file_path)
            df = pd.read_csv(full_path)
            
            # Extract region and model from filename
            # Remove simulation_results_ prefix and step suffix
            clean_name = file_path.replace("simulation_results_", "")
            # Find the step pattern and remove everything from there
            for step in ["_step1", "_step2", "_step3", "_step4"]:
                if step in clean_name:
                    clean_name = clean_name.split(step)[0]
                    break
            parts = clean_name.split("_")
            region = parts[0]
            model = "_".join(parts[1:])  # Model name might contain underscores
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot GPP data only
            ax.plot(df['year'], df['gpp_data'], 'b-', linewidth=2, label='GPP Data', alpha=0.8)
            ax.plot(df['year'], df['GPP'], 'b-', linewidth=1, label='GPP Model', alpha=0.6)
            
            # Customize plot
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
            ax.set_title(f'Step 2 Results: {region} / {model}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add MSE information if available
            if 'final_mse' in df.columns:
                mse = df['final_mse'].iloc[0]
                ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"Step 2 book saved to: {pdf_path}")

def create_step3_book(output_dir):
    """
    Create PDF book showing Step 3 results.
    
    Each panel shows:
    - GPP_data (blue line, linewidth=2)
    - GPP_model (fine blue line)
    """
    print("Creating Step 3 results book...")
    
    # Find Step 3 simulation results files
    step3_files = []
    if os.path.isdir(output_dir):
        step3_files = [f for f in os.listdir(output_dir) if f.startswith("simulation_results_") and "_step3" in f]
    
    if not step3_files:
        print("No Step 3 simulation results found.")
        return
    
    # Create PDF
    pdf_path = os.path.join(output_dir, "Step3_Results_Book.pdf")
    with PdfPages(pdf_path) as pdf:
        for file_path in sorted(step3_files):
            # Load simulation results
            full_path = os.path.join(output_dir, file_path)
            df = pd.read_csv(full_path)
            
            # Extract region and model from filename
            # Remove simulation_results_ prefix and step suffix
            clean_name = file_path.replace("simulation_results_", "")
            # Find the step pattern and remove everything from there
            for step in ["_step1", "_step2", "_step3", "_step4"]:
                if step in clean_name:
                    clean_name = clean_name.split(step)[0]
                    break
            parts = clean_name.split("_")
            region = parts[0]
            model = "_".join(parts[1:])  # Model name might contain underscores
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot GPP data only
            ax.plot(df['year'], df['gpp_data'], 'b-', linewidth=2, label='GPP Data', alpha=0.8)
            ax.plot(df['year'], df['GPP'], 'b-', linewidth=1, label='GPP Model', alpha=0.6)
            
            # Customize plot
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
            ax.set_title(f'Step 3 Results: {region} / {model}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add MSE information if available
            if 'final_mse' in df.columns:
                mse = df['final_mse'].iloc[0]
                ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"Step 3 book saved to: {pdf_path}")

def create_step4_book(output_dir):
    """
    Create PDF book showing Step 4 results.
    
    Each panel shows:
    - GPP_data (red line, linewidth=2)
    - GPP_model (fine red line)
    """
    print("Creating Step 4 results book...")
    
    # Find Step 4 simulation results files
    step4_files = []
    if os.path.isdir(output_dir):
        step4_files = [f for f in os.listdir(output_dir) if f.startswith("simulation_results_") and "_step4" in f]
    
    if not step4_files:
        print("No Step 4 simulation results found.")
        return
    
    # Create PDF
    pdf_path = os.path.join(output_dir, "Step4_Results_Book.pdf")
    with PdfPages(pdf_path) as pdf:
        for file_path in sorted(step4_files):
            # Load simulation results
            full_path = os.path.join(output_dir, file_path)
            df = pd.read_csv(full_path)
            
            # Extract region and model from filename
            # Remove simulation_results_ prefix and step suffix
            clean_name = file_path.replace("simulation_results_", "")
            # Find the step pattern and remove everything from there
            for step in ["_step1", "_step2", "_step3", "_step4"]:
                if step in clean_name:
                    clean_name = clean_name.split(step)[0]
                    break
            parts = clean_name.split("_")
            region = parts[0]
            model = "_".join(parts[1:])  # Model name might contain underscores
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot GPP data only
            ax.plot(df['year'], df['gpp_data'], 'r-', linewidth=2, label='GPP Data', alpha=0.8)
            ax.plot(df['year'], df['GPP'], 'r-', linewidth=1, label='GPP Model', alpha=0.6)
            
            # Customize plot
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
            ax.set_title(f'Step 4 Results: {region} / {model}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add MSE information if available
            if 'final_mse' in df.columns:
                mse = df['final_mse'].iloc[0]
                ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"Step 4 book saved to: {pdf_path}")

def create_step3_vs_step4_book(output_dir):
    """
    Create PDF book comparing Step 3 vs Step 4 results.
    
    Each panel shows:
    - GPP_data_Step3 (blue line, linewidth=2)
    - GPP_data_Step4 (red line, linewidth=2)
    - GPP_model_Step3 (fine blue line)
    - GPP_model_Step4 (fine red line)
    """
    print("Creating Step 3 vs Step 4 comparison book...")
    
    # Find Step 3 and Step 4 simulation results files
    step3_files = []
    step4_files = []
    
    if os.path.isdir(output_dir):
        step3_files = [f for f in os.listdir(output_dir) if f.startswith("simulation_results_") and "_step3" in f]
        step4_files = [f for f in os.listdir(output_dir) if f.startswith("simulation_results_") and "_step4" in f]
    
    if not step3_files or not step4_files:
        print("Step 3 or Step 4 simulation results not found.")
        return
    
    # Create mapping of region/model to file paths
    step3_map = {}
    step4_map = {}
    
    for file_path in step3_files:
        # Remove simulation_results_ prefix and step suffix
        clean_name = file_path.replace("simulation_results_", "")
        # Find the step pattern and remove everything from there
        for step in ["_step1", "_step2", "_step3", "_step4"]:
            if step in clean_name:
                clean_name = clean_name.split(step)[0]
                break
        parts = clean_name.split("_")
        region = parts[0]
        model = "_".join(parts[1:])  # Model name might contain underscores
        step3_map[(region, model)] = file_path
    
    for file_path in step4_files:
        # Remove simulation_results_ prefix and step suffix
        clean_name = file_path.replace("simulation_results_", "")
        # Find the step pattern and remove everything from there
        for step in ["_step1", "_step2", "_step3", "_step4"]:
            if step in clean_name:
                clean_name = clean_name.split(step)[0]
                break
        parts = clean_name.split("_")
        region = parts[0]
        model = "_".join(parts[1:])  # Model name might contain underscores
        step4_map[(region, model)] = file_path
    
    # Create PDF
    pdf_path = os.path.join(output_dir, "Step3_vs_Step4_Comparison_Book.pdf")
    with PdfPages(pdf_path) as pdf:
        # Find all unique region/model combinations
        all_combinations = set(step3_map.keys()) | set(step4_map.keys())
        
        for region, model in sorted(all_combinations):
            # Check if we have both Step 3 and Step 4 results
            if (region, model) not in step3_map or (region, model) not in step4_map:
                print(f"Warning: Missing Step 3 or Step 4 results for {region} / {model}")
                continue
            
            # Load both datasets
            step3_path = os.path.join(output_dir, step3_map[(region, model)])
            step4_path = os.path.join(output_dir, step4_map[(region, model)])
            
            df3 = pd.read_csv(step3_path)
            df4 = pd.read_csv(step4_path)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot GPP data and models
            ax.plot(df3['year'], df3['gpp_data'], 'b-', linewidth=2, label='GPP Data (Step 3)', alpha=0.8)
            ax.plot(df4['year'], df4['gpp_data'], 'r-', linewidth=2, label='GPP Data (Step 4)', alpha=0.8)
            ax.plot(df3['year'], df3['GPP'], 'b-', linewidth=1, label='GPP Model (Step 3)', alpha=0.6)
            ax.plot(df4['year'], df4['GPP'], 'r-', linewidth=1, label='GPP Model (Step 4)', alpha=0.6)
            
            # Customize plot
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('GPP (kg C m⁻² yr⁻¹)', fontsize=12)
            ax.set_title(f'Step 3 vs Step 4 Comparison: {region} / {model}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add MSE information if available
            if 'final_mse' in df3.columns and 'final_mse' in df4.columns:
                mse3 = df3['final_mse'].iloc[0]
                mse4 = df4['final_mse'].iloc[0]
                ax.text(0.02, 0.98, f'Step 3 MSE: {mse3:.4f}\nStep 4 MSE: {mse4:.4f}', 
                       transform=ax.transAxes, verticalalignment='top', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"Step 3 vs Step 4 comparison book saved to: {pdf_path}")

def create_all_books():
    """
    Create all three PDF books for the current run.
    """
    # Use the most recent output directory
    from step_utils import get_most_recent_output_directory
    output_dir = get_most_recent_output_directory()
    
    if output_dir is None:
        print("No output directories found. Run the analysis first.")
        return
    
    print(f"Creating PDF books in: {output_dir}")
    
    create_step1_book(output_dir)
    create_step2_book(output_dir)
    create_step3_book(output_dir)
    create_step4_book(output_dir)
    create_step3_vs_step4_book(output_dir)
    
    print("All PDF books created successfully!")
