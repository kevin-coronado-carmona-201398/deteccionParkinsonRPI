#!/usr/bin/env python3
"""
Script to prepare training data from dataset (3) folder structure
This script copies and renames files to match the expected naming convention
"""

import os
import shutil
from pathlib import Path

def prepare_training_data(source_base="dataset (3)/dataset/new folder", 
                         output_folder="training_data"):
    """
    Prepare training data by copying and renaming files
    
    Args:
        source_base: Base path containing patients/ and non_patients/ folders
        output_folder: Output folder where prepared data will be saved
    """
    print(f"[INFO] Preparing training data from {source_base}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Created output folder: {output_folder}")
    
    # Process patient files
    patients_folder = os.path.join(source_base, "patients")
    if os.path.exists(patients_folder):
        print(f"\n[INFO] Processing patient files from {patients_folder}")
        patient_files = [f for f in os.listdir(patients_folder) if f.endswith('.csv')]
        print(f"[INFO] Found {len(patient_files)} patient files")
        
        for idx, file in enumerate(patient_files, start=1):
            src = os.path.join(patients_folder, file)
            # Rename with P_ prefix and number
            dst = os.path.join(output_folder, f"P_{idx:03d}_{file}")
            shutil.copy2(src, dst)
            print(f"  Copied: {file} -> P_{idx:03d}_{file}")
    else:
        print(f"[WARNING] Patients folder not found: {patients_folder}")
    
    # Process non-patient files
    non_patients_folder = os.path.join(source_base, "non_patients")
    if os.path.exists(non_patients_folder):
        print(f"\n[INFO] Processing non-patient files from {non_patients_folder}")
        non_patient_files = [f for f in os.listdir(non_patients_folder) if f.endswith('.csv')]
        print(f"[INFO] Found {len(non_patient_files)} non-patient files")
        
        for idx, file in enumerate(non_patient_files, start=1):
            src = os.path.join(non_patients_folder, file)
            # Rename with N_ prefix and number
            dst = os.path.join(output_folder, f"N_{idx:03d}_{file}")
            shutil.copy2(src, dst)
            print(f"  Copied: {file} -> N_{idx:03d}_{file}")
    else:
        print(f"[WARNING] Non-patients folder not found: {non_patients_folder}")
    
    # Count final files
    final_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]
    p_files = [f for f in final_files if f.startswith('P_')]
    n_files = [f for f in final_files if f.startswith('N_')]
    
    print(f"\n[SUCCESS] Training data preparation complete!")
    print(f"  Total files: {len(final_files)}")
    print(f"  Patient files (P_): {len(p_files)}")
    print(f"  Non-patient files (N_): {len(n_files)}")
    print(f"  Output folder: {os.path.abspath(output_folder)}")
    
    return output_folder

if __name__ == "__main__":
    import sys
    
    # Allow custom paths via command line
    source_base = sys.argv[1] if len(sys.argv) > 1 else "dataset (3)/dataset/new folder"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "training_data"
    
    prepare_training_data(source_base, output_folder)

