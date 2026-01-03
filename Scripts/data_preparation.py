#!/usr/bin/env python3
"""
Athletic Training Data Preparation Script

This script loads, cleans, and processes athletic training data from Excel files.
It handles both treatment logs and injury reports, standardizing formats and
exporting cleaned datasets for analysis.

Author: Advanced Data Analytics Student
Date: 2025
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Optional


class ATDataProcessor:
    """Athletic Training Data Processor for cleaning and standardizing data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the data directory containing Excel files
        """
        self.data_path = Path(data_path)
        self.treatment_raw = None
        self.injury_raw = None
        self.treatment_clean = None
        self.injury_clean = None
    
    def load_raw_data(self, treatment_file: str, injury_file: str) -> None:
        """
        Load raw data from Excel files.
        
        Args:
            treatment_file: Name of treatment log Excel file
            injury_file: Name of injury report Excel file
        """
        treatment_path = self.data_path / treatment_file
        injury_path = self.data_path / injury_file
        
        print(f"Loading treatment data from: {treatment_path}")
        print(f"Loading injury data from: {injury_path}")
        
        self.treatment_raw = pd.read_excel(treatment_path)
        self.injury_raw = pd.read_excel(injury_path)
        
        print(f"Treatment raw shape: {self.treatment_raw.shape}")
        print(f"Injury raw shape: {self.injury_raw.shape}")
    
    def clean_treatment_data(self) -> pd.DataFrame:
        """
        Clean and standardize treatment data.
        
        Returns:
            Cleaned treatment DataFrame
        """
        if self.treatment_raw is None:
            raise ValueError("Raw treatment data not loaded. Call load_raw_data() first.")
        
        # Fix headers - first row contains column names
        df = self.treatment_raw.copy()
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Date_only'] = df['Date'].dt.date
            
            # Create time-based derived variables
            df['Hour_of_Day'] = df['Date'].dt.hour
            df['Day_of_Week'] = df['Date'].dt.day_name()
            df['Day_of_Week_Num'] = df['Date'].dt.dayofweek  # 0=Monday
            df['Week_of_Year'] = df['Date'].dt.isocalendar().week
            df['Month'] = df['Date'].dt.month
            df['Is_Weekend'] = df['Day_of_Week_Num'].isin([5, 6])  # Sat, Sun
            
            # Time blocks for analysis
            df['Time_Block'] = pd.cut(df['Hour_of_Day'], 
                                    bins=[0, 8, 12, 17, 24], 
                                    labels=['Early', 'Morning', 'Afternoon', 'Evening'],
                                    include_lowest=True)
        
        # Clean categorical columns
        categorical_cols = ['Body Part', 'Service', 'Treating Provider', 'Missed', 'Scheduled']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Create binary flags
        if 'Missed' in df.columns:
            df['Missed_Flag'] = df['Missed'].str.lower().eq('yes')
        
        if 'Scheduled' in df.columns:
            df['Scheduled_Flag'] = df['Scheduled'].str.lower().eq('yes')
            df['Walk_In'] = ~df['Scheduled_Flag']  # Opposite of scheduled
        
        # Body part categorization
        if 'Body Part' in df.columns:
            upper_extremity = ['Shoulder', 'Elbow', 'Wrist', 'Hand', 'Finger', 'Arm', 'Forearm']
            lower_extremity = ['Hip', 'Knee', 'Ankle', 'Foot', 'Toe', 'Thigh', 'Calf', 'Shin']
            spine = ['Neck', 'Back', 'Spine', 'Cervical', 'Thoracic', 'Lumbar']
            
            df['Body_Region'] = df['Body Part'].apply(lambda x: 
                'Upper Extremity' if any(part in str(x) for part in upper_extremity) else
                'Lower Extremity' if any(part in str(x) for part in lower_extremity) else
                'Spine/Core' if any(part in str(x) for part in spine) else
                'Other'
            )
        
        # Provider workload metrics (calculated per day)
        if 'Treating Provider' in df.columns and 'Date_only' in df.columns:
            daily_provider_counts = df.groupby(['Date_only', 'Treating Provider']).size().reset_index(name='Daily_Provider_Load')
            df = df.merge(daily_provider_counts, on=['Date_only', 'Treating Provider'], how='left')
        
        self.treatment_clean = df
        print(f"Treatment data cleaned. Final shape: {df.shape}")
        return df
    
    def clean_injury_data(self) -> pd.DataFrame:
        """
        Clean and standardize injury data.
        
        Returns:
            Cleaned injury DataFrame
        """
        if self.injury_raw is None:
            raise ValueError("Raw injury data not loaded. Call load_raw_data() first.")
        
        # Fix headers - first row contains column names
        df = self.injury_raw.copy()
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        
        # Convert date columns
        date_cols = [
            'Problem Date',
            'Expected Return Date', 
            'Actual Return Date',
            'Reported Date',
            'Problem Created Date Time'
        ]
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        if 'Days Out' in df.columns:
            df['Days Out'] = pd.to_numeric(df['Days Out'], errors='coerce')
            
            # Create injury severity categories
            df['Injury_Severity'] = pd.cut(df['Days Out'], 
                                         bins=[-1, 0, 7, 21, float('inf')], 
                                         labels=['No Time Lost', 'Mild (1-7 days)', 'Moderate (8-21 days)', 'Severe (22+ days)'],
                                         include_lowest=True)
            
            df['Is_Chronic'] = df['Days Out'] > 30
            df['Quick_Recovery'] = df['Days Out'] <= 7
        
        # Calculate recovery metrics
        if 'Problem Date' in df.columns and 'Actual Return Date' in df.columns:
            df['Recovery_Days_Calculated'] = (df['Actual Return Date'] - df['Problem Date']).dt.days
            
            if 'Expected Return Date' in df.columns:
                df['Recovery_Accuracy'] = (df['Actual Return Date'] - df['Expected Return Date']).dt.days
                df['Exceeded_Expected'] = df['Recovery_Accuracy'] > 0
                df['Return_Success'] = df['Actual Return Date'].notna()
        
        # Time to report injury
        if 'Problem Date' in df.columns and 'Reported Date' in df.columns:
            df['Days_to_Report'] = (df['Reported Date'] - df['Problem Date']).dt.days
            df['Immediate_Report'] = df['Days_to_Report'] <= 1
        
        # Academic year calculation
        if 'Graduation Year' in df.columns and 'Problem Date' in df.columns:
            df['Academic_Year'] = df['Graduation Year'] - (df['Problem Date'].dt.year - 2021)  # Adjust base year as needed
            df['Class_Standing'] = df['Academic_Year'].map({
                1: 'Freshman', 2: 'Sophomore', 3: 'Junior', 4: 'Senior'
            })
        
        # Body part categorization (same as treatment)
        if 'Body Part' in df.columns:
            upper_extremity = ['Shoulder', 'Elbow', 'Wrist', 'Hand', 'Finger', 'Arm', 'Forearm']
            lower_extremity = ['Hip', 'Knee', 'Ankle', 'Foot', 'Toe', 'Thigh', 'Calf', 'Shin']
            spine = ['Neck', 'Back', 'Spine', 'Cervical', 'Thoracic', 'Lumbar']
            
            df['Body_Region'] = df['Body Part'].apply(lambda x: 
                'Upper Extremity' if any(part in str(x) for part in upper_extremity) else
                'Lower Extremity' if any(part in str(x) for part in lower_extremity) else
                'Spine/Core' if any(part in str(x) for part in spine) else
                'Other'
            )
        
        # Clean categorical columns
        categorical_cols = ['Body Part', 'Condition', 'Problem Status', 'Datalys Injury Type']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        self.injury_clean = df
        print(f"Injury data cleaned. Final shape: {df.shape}")
        return df
    
    def export_cleaned_data(self, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Export cleaned datasets to Excel files.
        
        Args:
            output_dir: Directory to save files (defaults to data_path)
            
        Returns:
            Tuple of (treatment_file_path, injury_file_path)
        """
        if self.treatment_clean is None or self.injury_clean is None:
            raise ValueError("Data not cleaned. Call clean_treatment_data() and clean_injury_data() first.")
        
        if output_dir is None:
            output_dir = self.data_path
        else:
            output_dir = Path(output_dir)
        
        treatment_output = output_dir / "Treatment_Cleaned.xlsx"
        injury_output = output_dir / "Injury_Cleaned.xlsx"
        
        self.treatment_clean.to_excel(treatment_output, index=False)
        self.injury_clean.to_excel(injury_output, index=False)
        
        print(f"Treatment data exported to: {treatment_output}")
        print(f"Injury data exported to: {injury_output}")
        
        return str(treatment_output), str(injury_output)
    
    def get_data_summary(self) -> dict:
        """
        Generate summary statistics for cleaned data.
        
        Returns:
            Dictionary containing summary statistics
        """
        summary = {}
        
        if self.treatment_clean is not None:
            df = self.treatment_clean
            summary['treatment'] = {
                'total_records': len(df),
                'date_range': (df['Date'].min(), df['Date'].max()) if 'Date' in df.columns else None,
                'unique_providers': df['Treating Provider'].nunique() if 'Treating Provider' in df.columns else None,
                'unique_services': df['Service'].nunique() if 'Service' in df.columns else None,
                'unique_body_parts': df['Body Part'].nunique() if 'Body Part' in df.columns else None
            }
        
        if self.injury_clean is not None:
            df = self.injury_clean
            summary['injury'] = {
                'total_records': len(df),
                'date_range': (df['Problem Date'].min(), df['Problem Date'].max()) if 'Problem Date' in df.columns else None,
                'unique_conditions': df['Condition'].nunique() if 'Condition' in df.columns else None,
                'unique_body_parts': df['Body Part'].nunique() if 'Body Part' in df.columns else None
            }
        
        return summary


def main():
    """Main execution function."""
    # Configuration
    DATA_PATH = "/Users/philip_koutsaftis/Library/CloudStorage/GoogleDrive-philipkoutsaftis@gmail.com/My Drive/AT_Dept_Data/Data/Aug-Nov17"
    TREATMENT_FILE = "Treatment Log Aug thru Nov 17 (1).xlsx"
    INJURY_FILE = "Injury Report aug thru 11_17.xlsx"
    
    # Initialize processor
    processor = ATDataProcessor(DATA_PATH)
    
    try:
        # Load raw data
        processor.load_raw_data(TREATMENT_FILE, INJURY_FILE)
        
        # Clean data
        treatment_df = processor.clean_treatment_data()
        injury_df = processor.clean_injury_data()
        
        # Export cleaned data
        treatment_path, injury_path = processor.export_cleaned_data()
        
        # Print summary
        summary = processor.get_data_summary()
        print("\n" + "="*50)
        print("DATA PROCESSING SUMMARY")
        print("="*50)
        
        if 'treatment' in summary:
            t_sum = summary['treatment']
            print(f"Treatment Data:")
            print(f"  - Records: {t_sum['total_records']:,}")
            print(f"  - Date Range: {t_sum['date_range']}")
            print(f"  - Providers: {t_sum['unique_providers']}")
            print(f"  - Services: {t_sum['unique_services']}")
            print(f"  - Body Parts: {t_sum['unique_body_parts']}")
        
        if 'injury' in summary:
            i_sum = summary['injury']
            print(f"\nInjury Data:")
            print(f"  - Records: {i_sum['total_records']:,}")
            print(f"  - Date Range: {i_sum['date_range']}")
            print(f"  - Conditions: {i_sum['unique_conditions']}")
            print(f"  - Body Parts: {i_sum['unique_body_parts']}")
        
        print(f"\nCleaned files saved successfully!")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()