"""
Feature Engineering Module for Car Price Prediction
Extracts brand equity, vehicle classification, brand goodwill scores,
technical automotive specifications (horsepower, mileage, engine displacement),
and derived usage dynamics (vehicle age, km per year).
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Domain Keywords for Two-Wheelers present in the dataset
TWO_WHEELER_KEYWORDS = [
    'activa', 'bajaj', 'hero', 'yamaha', 'tvs', 'ktm', 'royal', 
    'hyosung', 'mahindra mojo', 'um renegade', 'suzuki access', 'jupiter', 'wego'
]

# Established Indian Automotive Market Brand Goodwill Ratings (Scale: 1.0 - 10.0)
# Reflects customer brand perception, reliability ranking, and historical resale value retention
BRAND_GOODWILL_MAP = {
    'Toyota': 9.5,        # Highest resale retention in India (Fortuner, Innova legendary resale)
    'Maruti Suzuki': 9.2, # Unmatched service network, parts availability, low depreciation
    'Honda': 8.8,         # High engine longevity and trusted premium commuter status
    'Hyundai': 8.6,       # Modern features, strong resale across hatchbacks & compact SUVs
    'Royal Enfield': 8.7, # Cult classic motorcycle status with strong second-hand value
    'KTM': 8.2,           # High-performance enthusiast segment, good retention
    'Yamaha': 8.0,        # High build quality & engine performance
    'Bajaj': 7.6,         # Broad commuter & sports bike appeal (Pulsar family)
    'Hero': 7.5,          # High-volume mass market utility, low running costs
    'TVS': 7.3,           # Strong performance commuter & scooter market presence
    'Suzuki': 7.4,        # Trusted Japanese mechanicals (Access scooter)
    'Mahindra': 7.5,      # Rugged utility brand
    'Hyosung': 6.0,       # Niche niche Korean sports bike, limited service presence
    'UM': 5.8             # Niche cruiser, limited service footprint
}


def extract_brand_name(car_name):
    """
    Map Car_Name to its parent automotive brand / manufacturer.
    
    Parameters:
        car_name (str): The name/model string.
        
    Returns:
        str: Manufacturer / Brand name.
    """
    clean_name = str(car_name).strip().lower()
    first_word = clean_name.split()[0]
    
    # Maruti Suzuki models
    maruti_models = [
        'ritz', 'sx4', 'ciaz', 'wagon', 'swift', 'vitara', 's', 'alto', 
        'ertiga', 'dzire', 'ignis', '800', 'baleno', 'omni'
    ]
    # Toyota models
    toyota_models = ['fortuner', 'innova', 'corolla', 'etios', 'land', 'camry']
    # Honda models
    honda_models = ['city', 'brio', 'amaze', 'jazz', 'activa', 'honda']
    # Hyundai models
    hyundai_models = ['i20', 'verna', 'i10', 'grand', 'creta', 'elantra', 'xcent', 'eon']
    
    if any(m == first_word for m in maruti_models):
        return 'Maruti Suzuki'
    if any(m == first_word for m in toyota_models):
        return 'Toyota'
    if any(m == first_word or first_word.startswith('activa') for m in honda_models):
        return 'Honda'
    if any(m == first_word for m in hyundai_models):
        return 'Hyundai'
    if first_word == 'royal':
        return 'Royal Enfield'
    if first_word in ['bajaj', 'hero', 'yamaha', 'tvs', 'ktm', 'hyosung', 'mahindra', 'um', 'suzuki']:
        return first_word.capitalize()
        
    return 'Other'


def classify_vehicle_type(car_name):
    """
    Classify whether a record is a 4-wheel Passenger Car or a Two-Wheeler.
    """
    name_lower = str(car_name).lower()
    if any(keyword in name_lower for keyword in TWO_WHEELER_KEYWORDS):
        return 'Two-Wheeler'
    return 'Car'


def lookup_technical_specs(car_name, vehicle_type, present_price):
    """
    Enrich data with technical specifications: Horsepower (BHP), Mileage (kmpl),
    and Engine Displacement (cc) based on vehicle model domain knowledge.
    
    Parameters:
        car_name (str): Vehicle model name.
        vehicle_type (str): 'Car' or 'Two-Wheeler'.
        present_price (float): Ex-showroom price in Lakhs.
        
    Returns:
        tuple: (hp_bhp, mileage_kmpl, engine_cc)
    """
    name = str(car_name).lower()
    
    if vehicle_type == 'Two-Wheeler':
        if '390' in name or 'rc390' in name:
            return 43.0, 26.0, 373
        if 'rc200' in name or 'ns 200' in name or 'rs200' in name:
            return 24.5, 35.0, 199
        if 'classic 500' in name or 'thunder 500' in name:
            return 27.2, 30.0, 499
        if 'classic 350' in name or 'thunder 350' in name or 'bullet 350' in name:
            return 19.8, 35.0, 346
        if 'dominar' in name or 'mojo' in name:
            return 35.0, 28.0, 373
        if 'gt250r' in name:
            return 28.0, 25.0, 249
        if 'renegade' in name:
            return 25.0, 30.0, 279
        if '220' in name:
            return 20.6, 38.0, 220
        if any(tag in name for tag in ['150', '160', 'fz', 'fazer', 'hornet', 'xtreme', 'hunk', 'unicorn', 'trigger']):
            return 14.0, 45.0, 150
        if any(tag in name for tag in ['125', 'shine', 'glamour', 'ignitor', 'access']):
            return 10.5, 55.0, 125
        if any(tag in name for tag in ['activa', 'jupiter', 'wego']):
            return 8.0, 58.0, 110
        # Standard commuter bike default (100cc)
        return 8.0, 65.0, 100
        
    # Passenger Cars
    if 'land cruiser' in name:
        return 261.0, 9.0, 4461
    if 'fortuner' in name:
        return 174.0, 14.2, 2755
    if 'camry' in name:
        return 158.0, 19.1, 2494
    if 'innova' in name:
        return 101.0, 12.9, 2494
    if 'elantra' in name:
        return 150.0, 14.6, 1999
    if 'corolla' in name:
        return 138.0, 14.3, 1798
    if 'creta' in name or 'verna' in name:
        return 121.0, 17.0, 1591
    if 'city' in name:
        return 117.0, 17.8, 1497
    if 'sx4' in name:
        return 103.0, 15.5, 1586
    if 'ciaz' in name or 'ertiga' in name:
        return 91.0, 20.0, 1373
    if 's cross' in name or 'vitara' in name:
        return 89.0, 23.6, 1248
    if 'amaze' in name or 'jazz' in name or 'brio' in name:
        return 88.0, 18.2, 1198
    if any(m in name for m in ['swift', 'dzire', 'baleno', 'ritz', 'ignis', 'i20', 'grand i10', 'xcent', 'etios']):
        return 83.0, 20.5, 1197
    if 'i10' in name:
        return 68.0, 19.8, 1086
    if 'wagon r' in name or 'alto k10' in name:
        return 67.0, 22.0, 998
    if 'eon' in name:
        return 55.0, 21.1, 814
    if 'alto 800' in name:
        return 47.0, 22.7, 796
    if 'omni' in name or '800' in name:
        return 35.0, 16.5, 796
        
    # Price-based fallback heuristic for cars
    if present_price > 25.0:
        return 190.0, 12.0, 2800
    elif present_price > 12.0:
        return 130.0, 15.5, 1800
    elif present_price > 6.0:
        return 85.0, 19.5, 1200
    else:
        return 65.0, 22.0, 1000


def engineer_car_features(df, reference_year=2018):
    """
    Perform full feature engineering pipeline on the raw car dataset.
    
    Parameters:
        df (pd.DataFrame): Raw or cleaned car dataframe.
        reference_year (int): Reference baseline year for calculating vehicle age.
        
    Returns:
        pd.DataFrame: Dataframe with all engineered features.
    """
    df_eng = df.copy()
    
    # 1. Brand extraction & Vehicle Classification
    df_eng['Brand'] = df_eng['Car_Name'].apply(extract_brand_name)
    df_eng['Vehicle_Type'] = df_eng['Car_Name'].apply(classify_vehicle_type)
    
    # 2. Brand Goodwill Score
    df_eng['Brand_Goodwill'] = df_eng['Brand'].map(BRAND_GOODWILL_MAP).fillna(7.0)
    
    # 3. Age & Annual Usage Intensity
    df_eng['Vehicle_Age'] = reference_year - df_eng['Year']
    # Prevent divide by zero if age is 0
    df_eng['Km_per_Year'] = df_eng['Driven_kms'] / (df_eng['Vehicle_Age'] + 1)
    
    # 4. Technical Specs Lookup (Horsepower, Mileage, Engine CC)
    specs = df_eng.apply(
        lambda row: lookup_technical_specs(row['Car_Name'], row['Vehicle_Type'], row['Present_Price']),
        axis=1
    )
    df_eng['Horsepower_bhp'] = [s[0] for s in specs]
    df_eng['Mileage_kmpl'] = [s[1] for s in specs]
    df_eng['Engine_Capacity_cc'] = [s[2] for s in specs]
    
    # 5. Depreciation Analysis (for EDA & Diagnostics)
    if 'Selling_Price' in df_eng.columns:
        df_eng['Price_Depreciation'] = df_eng['Present_Price'] - df_eng['Selling_Price']
        df_eng['Value_Retention_Ratio'] = df_eng['Selling_Price'] / df_eng['Present_Price']
        
    return df_eng


def get_preprocessor(categorical_features, numerical_features):
    """
    Create a scikit-learn ColumnTransformer for encoding categoricals and scaling numericals.
    
    Parameters:
        categorical_features (list): Names of categorical columns.
        numerical_features (list): Names of numerical columns.
        
    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


if __name__ == "__main__":
    from data_preprocessing import load_dataset, clean_dataset
    df = clean_dataset(load_dataset())
    engineered = engineer_car_features(df)
    print("\nSample Engineered Records:")
    print(engineered[['Car_Name', 'Brand', 'Vehicle_Type', 'Brand_Goodwill', 'Horsepower_bhp', 'Mileage_kmpl', 'Vehicle_Age', 'Km_per_Year', 'Selling_Price']].head())
