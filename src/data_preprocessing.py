"""
Data Preprocessing Module for Car Price Prediction
Handles dataset loading, validation, duplicate handling, and outlier diagnosis.
"""

import os
import pandas as pd
import numpy as np


def load_dataset(file_path="car data.csv"):
    """
    Load the car dataset from CSV file.
    
    Parameters:
        file_path (str): Relative or absolute path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def inspect_data_quality(df):
    """
    Check for missing values, duplicates, and column data types.
    
    Parameters:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        dict: Summary of data quality checks.
    """
    missing_counts = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing_counts[missing_counts > 0].to_dict(),
        "duplicate_rows": int(duplicate_count),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    print("\n--- Data Quality Inspection ---")
    print(f"Total rows: {quality_report['total_rows']}")
    print(f"Total columns: {quality_report['total_columns']}")
    print(f"Duplicate rows detected: {quality_report['duplicate_rows']}")
    
    if quality_report['missing_values']:
        print(f"[WARNING] Missing values detected: {quality_report['missing_values']}")
    else:
        print("[INFO] No missing values detected across all columns.")
        
    return quality_report


def clean_dataset(df, drop_duplicates=True):
    """
    Clean the dataset by removing duplicate records and resetting the index.
    
    Parameters:
        df (pd.DataFrame): Raw dataframe.
        drop_duplicates (bool): Whether to drop exact duplicate rows.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df_cleaned = df.copy()
    
    if drop_duplicates:
        initial_len = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
        dropped_count = initial_len - len(df_cleaned)
        if dropped_count > 0:
            print(f"[INFO] Dropped {dropped_count} duplicate row(s). Cleaned dataset shape: {df_cleaned.shape}")
            
    return df_cleaned


def detect_outliers_iqr(df, numerical_cols=None):
    """
    Identify potential outliers in numerical columns using the Interquartile Range (IQR) method.
    
    Parameters:
        df (pd.DataFrame): Dataframe to analyze.
        numerical_cols (list): List of column names to check.
        
    Returns:
        dict: Outlier boundaries and detected outlier counts.
    """
    if numerical_cols is None:
        numerical_cols = ['Selling_Price', 'Present_Price', 'Driven_kms']
        
    outlier_summary = {}
    for col in numerical_cols:
        if col in df.columns:
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outlier_summary[col] = {
                "q25": round(float(q25), 2),
                "q75": round(float(q75), 2),
                "iqr": round(float(iqr), 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "outlier_count": len(outliers),
                "outlier_percentage": round((len(outliers) / len(df)) * 100, 2)
            }
            
    return outlier_summary


if __name__ == "__main__":
    df = load_dataset()
    inspect_data_quality(df)
    df_clean = clean_dataset(df)
    outliers = detect_outliers_iqr(df_clean)
    print("\nOutlier Summary:")
    for col, res in outliers.items():
        print(f"  {col}: {res['outlier_count']} outliers ({res['outlier_percentage']}%) [Boundaries: {res['lower_bound']} to {res['upper_bound']}]")
