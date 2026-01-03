#!/usr/bin/env python3
"""
Athletic Training Data Merger Script

Merges new period data into master historical datasets.
Handles deduplication and maintains data integrity.

Author: Advanced Data Analytics Student
Date: 2025
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime


class ATDataMerger:
    """Merges new data into master historical datasets."""
    
    def __init__(self, master_dir: str):
        """
        Initialize the data merger.
        
        Args:
            master_dir: Path to master data directory
        """
        self.master_dir = Path(master_dir)
        self.master_dir.mkdir(parents=True, exist_ok=True)
        
        self.treatment_master_path = self.master_dir / "Treatment_Master.xlsx"
        self.injury_master_path = self.master_dir / "Injury_Master.xlsx"
        self.encounter_master_path = self.master_dir / "Encounter_Master.xlsx"
    
    def load_or_create_master(self, master_path: Path) -> pd.DataFrame:
        """Load existing master file or create empty DataFrame."""
        if master_path.exists():
            print(f"Loading existing master: {master_path}")
            return pd.read_excel(master_path)
        else:
            print(f"No master file found. Will create new: {master_path}")
            return pd.DataFrame()
    
    def merge_treatment_data(self, new_data_path: str) -> pd.DataFrame:
        """
        Merge new treatment data into master.
        
        Args:
            new_data_path: Path to new treatment data file
            
        Returns:
            Updated master DataFrame
        """
        # Load master
        master_df = self.load_or_create_master(self.treatment_master_path)
        
        # Load new data
        new_df = pd.read_excel(new_data_path)
        print(f"New treatment data shape: {new_df.shape}")
        
        if master_df.empty:
            merged_df = new_df
        else:
            # Concatenate
            merged_df = pd.concat([master_df, new_df], ignore_index=True)
            
            # Deduplicate based on Date, Patient DOB, Service, Body Part
            if 'Date' in merged_df.columns:
                dedup_cols = ['Date', 'Patient DOB', 'Service', 'Body Part']
                dedup_cols = [c for c in dedup_cols if c in merged_df.columns]
                
                before_count = len(merged_df)
                merged_df = merged_df.drop_duplicates(subset=dedup_cols, keep='first')
                after_count = len(merged_df)
                
                print(f"Removed {before_count - after_count} duplicate treatment records")
        
        # Sort by date
        if 'Date' in merged_df.columns:
            merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')
            merged_df = merged_df.sort_values('Date').reset_index(drop=True)
        
        print(f"Master treatment data shape: {merged_df.shape}")
        return merged_df
    
    def merge_injury_data(self, new_data_path: str) -> pd.DataFrame:
        """
        Merge new injury data into master.
        
        Args:
            new_data_path: Path to new injury data file
            
        Returns:
            Updated master DataFrame
        """
        # Load master
        master_df = self.load_or_create_master(self.injury_master_path)
        
        # Load new data
        new_df = pd.read_excel(new_data_path)
        print(f"New injury data shape: {new_df.shape}")
        
        if master_df.empty:
            merged_df = new_df
        else:
            # Concatenate
            merged_df = pd.concat([master_df, new_df], ignore_index=True)
            
            # Deduplicate based on Problem Date, Body Part, Condition
            if 'Problem Date' in merged_df.columns:
                dedup_cols = ['Problem Date', 'Body Part', 'Condition', 'Graduation Year']
                dedup_cols = [c for c in dedup_cols if c in merged_df.columns]
                
                before_count = len(merged_df)
                merged_df = merged_df.drop_duplicates(subset=dedup_cols, keep='first')
                after_count = len(merged_df)
                
                print(f"Removed {before_count - after_count} duplicate injury records")
        
        # Sort by date
        if 'Problem Date' in merged_df.columns:
            merged_df['Problem Date'] = pd.to_datetime(merged_df['Problem Date'], errors='coerce')
            merged_df = merged_df.sort_values('Problem Date').reset_index(drop=True)
        
        print(f"Master injury data shape: {merged_df.shape}")
        return merged_df
    
    def merge_encounter_data(self, new_data_path: str) -> pd.DataFrame:
        """
        Merge new encounter log data into master.
        
        Args:
            new_data_path: Path to new encounter log file
            
        Returns:
            Updated master DataFrame
        """
        # Load master
        master_df = self.load_or_create_master(self.encounter_master_path)
        
        # Load new data
        new_df = pd.read_excel(new_data_path)
        print(f"New encounter data shape: {new_df.shape}")
        
        if master_df.empty:
            merged_df = new_df
        else:
            # Concatenate
            merged_df = pd.concat([master_df, new_df], ignore_index=True)
            
            # Deduplicate based on Date, Service, Body Part, Treating Provider
            if 'Date' in merged_df.columns:
                dedup_cols = ['Date', 'Service', 'Body Part', 'Treating Provider']
                dedup_cols = [c for c in dedup_cols if c in merged_df.columns]
                
                before_count = len(merged_df)
                merged_df = merged_df.drop_duplicates(subset=dedup_cols, keep='first')
                after_count = len(merged_df)
                
                print(f"Removed {before_count - after_count} duplicate encounter records")
        
        # Sort by date
        if 'Date' in merged_df.columns:
            merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')
            merged_df = merged_df.sort_values('Date').reset_index(drop=True)
        
        print(f"Master encounter data shape: {merged_df.shape}")
        return merged_df
    
    def save_master_files(self, treatment_df: pd.DataFrame, injury_df: pd.DataFrame, encounter_df: pd.DataFrame = None):
        """Save updated master files."""
        treatment_df.to_excel(self.treatment_master_path, index=False)
        injury_df.to_excel(self.injury_master_path, index=False)
        
        if encounter_df is not None:
            encounter_df.to_excel(self.encounter_master_path, index=False)
        
        print(f"\n✅ Master files updated:")
        print(f"  - Treatment: {self.treatment_master_path}")
        print(f"  - Injury: {self.injury_master_path}")
        if encounter_df is not None:
            print(f"  - Encounter: {self.encounter_master_path}")
    
    def get_date_range_summary(self, treatment_df: pd.DataFrame, injury_df: pd.DataFrame, encounter_df: pd.DataFrame = None):
        """Print date range summary."""
        print("\n" + "="*50)
        print("MASTER DATA SUMMARY")
        print("="*50)
        
        if 'Date' in treatment_df.columns:
            treatment_df['Date'] = pd.to_datetime(treatment_df['Date'], errors='coerce')
            print(f"Treatment Data:")
            print(f"  - Total Records: {len(treatment_df):,}")
            print(f"  - Date Range: {treatment_df['Date'].min()} → {treatment_df['Date'].max()}")
            print(f"  - Time Span: {(treatment_df['Date'].max() - treatment_df['Date'].min()).days} days")
        
        if 'Problem Date' in injury_df.columns:
            injury_df['Problem Date'] = pd.to_datetime(injury_df['Problem Date'], errors='coerce')
            print(f"\nInjury Data:")
            print(f"  - Total Records: {len(injury_df):,}")
            print(f"  - Date Range: {injury_df['Problem Date'].min()} → {injury_df['Problem Date'].max()}")
            print(f"  - Time Span: {(injury_df['Problem Date'].max() - injury_df['Problem Date'].min()).days} days")
        
        if encounter_df is not None and 'Date' in encounter_df.columns:
            encounter_df['Date'] = pd.to_datetime(encounter_df['Date'], errors='coerce')
            print(f"\nEncounter Data:")
            print(f"  - Total Records: {len(encounter_df):,}")
            print(f"  - Date Range: {encounter_df['Date'].min()} → {encounter_df['Date'].max()}")
            print(f"  - Time Span: {(encounter_df['Date'].max() - encounter_df['Date'].min()).days} days")


def main():
    """Main execution function."""
    # Configuration
    BASE_PATH = "/Users/philip_koutsaftis/Library/CloudStorage/GoogleDrive-philipkoutsaftis@gmail.com/My Drive/AT_Dept_Data/Data"
    MASTER_DIR = f"{BASE_PATH}/Master"
    
    # New data to merge (cleaned files)
    NEW_TREATMENT = f"{BASE_PATH}/Aug-Nov17/Treatment_Cleaned.xlsx"
    NEW_INJURY = f"{BASE_PATH}/Aug-Nov17/Injury_Cleaned.xlsx"
    NEW_ENCOUNTER = f"{BASE_PATH}/Encounter Log Table-12_1_2025.xlsx"
    
    # Initialize merger
    merger = ATDataMerger(MASTER_DIR)
    
    try:
        print("Starting data merge process...\n")
        
        # Merge treatment data
        treatment_master = merger.merge_treatment_data(NEW_TREATMENT)
        
        # Merge injury data
        injury_master = merger.merge_injury_data(NEW_INJURY)
        
        # Merge encounter data
        encounter_master = merger.merge_encounter_data(NEW_ENCOUNTER)
        
        # Save master files
        merger.save_master_files(treatment_master, injury_master, encounter_master)
        
        # Print summary
        merger.get_date_range_summary(treatment_master, injury_master, encounter_master)
        
        print("\n✅ Data merge completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        raise


if __name__ == "__main__":
    main()
