"""
Inference and Prediction Module for Car Price Prediction
Allows single-record or batch pricing estimation via Python API or Command-Line Interface (CLI).
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

# Import feature engineering helpers
from feature_engineering import (
    extract_brand_name,
    classify_vehicle_type,
    lookup_technical_specs,
    BRAND_GOODWILL_MAP
)


class CarPricePredictor:
    """Car price prediction estimator wrapping the trained model pipeline."""
    
    def __init__(self, model_path="models/best_car_price_model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at: {model_path}. Please run model training first.")
        
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.feature_columns = artifact['feature_columns']
        
    def predict_single(self, car_name, year, present_price, driven_kms, 
                       fuel_type="Petrol", selling_type="Dealer", 
                       transmission="Manual", owner=0, reference_year=2018):
        """
        Predict selling price for a single vehicle record.
        
        Parameters:
            car_name (str): Model name (e.g. 'swift', 'city', 'fortuner').
            year (int): Year of manufacture / registration.
            present_price (float): Ex-showroom new vehicle price in Lakhs INR.
            driven_kms (int): Odometer reading in kilometers.
            fuel_type (str): 'Petrol', 'Diesel', or 'CNG'.
            selling_type (str): 'Dealer' or 'Individual'.
            transmission (str): 'Manual' or 'Automatic'.
            owner (int): Number of previous owners (0, 1, 3).
            reference_year (int): Baseline reference year.
            
        Returns:
            dict: Prediction results including estimated resale value, depreciation, and specs.
        """
        # 1. Feature Engineering
        brand = extract_brand_name(car_name)
        vehicle_type = classify_vehicle_type(car_name)
        goodwill = BRAND_GOODWILL_MAP.get(brand, 7.0)
        age = max(0, reference_year - year)
        km_per_year = driven_kms / (age + 1)
        
        hp_bhp, mileage_kmpl, engine_cc = lookup_technical_specs(car_name, vehicle_type, present_price)
        
        # 2. Build single-row DataFrame
        raw_dict = {
            'Present_Price': present_price,
            'Driven_kms': driven_kms,
            'Owner': owner,
            'Brand_Goodwill': goodwill,
            'Vehicle_Age': age,
            'Km_per_Year': km_per_year,
            'Horsepower_bhp': hp_bhp,
            'Mileage_kmpl': mileage_kmpl,
            'Engine_Capacity_cc': engine_cc,
            'Fuel_Type': fuel_type,
            'Selling_type': selling_type,
            'Transmission': transmission,
            'Vehicle_Type': vehicle_type,
            'Brand': brand
        }
        
        input_df = pd.DataFrame([raw_dict])
        
        # 3. Categorical encoding
        cat_cols = ['Fuel_Type', 'Selling_type', 'Transmission', 'Vehicle_Type', 'Brand']
        encoded_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
        
        # Align with model's expected columns
        for col in self.feature_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
                
        # Reorder columns to exact training sequence
        X_aligned = encoded_df[self.feature_columns]
        
        # 4. Predict
        predicted_price_lakhs = float(self.model.predict(X_aligned)[0])
        # Resale cannot exceed original price or be strictly negative
        predicted_price_lakhs = max(0.05, min(present_price * 1.05, predicted_price_lakhs))
        
        predicted_inr = int(predicted_price_lakhs * 100000)
        depreciation_lakhs = max(0.0, present_price - predicted_price_lakhs)
        retention_pct = round((predicted_price_lakhs / present_price) * 100, 2)
        
        return {
            "vehicle_name": car_name,
            "brand": brand,
            "vehicle_type": vehicle_type,
            "manufacturing_year": year,
            "vehicle_age_years": age,
            "present_price_lakhs": round(present_price, 2),
            "predicted_selling_price_lakhs": round(predicted_price_lakhs, 2),
            "predicted_price_inr": f"Rs. {predicted_inr:,}",
            "estimated_depreciation_lakhs": round(depreciation_lakhs, 2),
            "value_retention_percent": f"{retention_pct}%",
            "specifications": {
                "horsepower_bhp": hp_bhp,
                "mileage_kmpl": mileage_kmpl,
                "engine_capacity_cc": engine_cc,
                "brand_goodwill_score": goodwill
            }
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Car Resale Selling Price using Machine Learning")
    parser.add_argument("--car_name", type=str, default="honda city", help="Model name (e.g., 'swift', 'city', 'fortuner')")
    parser.add_argument("--year", type=int, default=2015, help="Registration/Manufacturing year (e.g. 2015)")
    parser.add_argument("--present_price", type=float, default=11.5, help="Ex-showroom price in Lakhs INR (e.g. 11.5)")
    parser.add_argument("--driven_kms", type=int, default=38000, help="Total distance driven in km (e.g. 38000)")
    parser.add_argument("--fuel_type", type=str, default="Petrol", choices=["Petrol", "Diesel", "CNG"])
    parser.add_argument("--selling_type", type=str, default="Dealer", choices=["Dealer", "Individual"])
    parser.add_argument("--transmission", type=str, default="Manual", choices=["Manual", "Automatic"])
    parser.add_argument("--owner", type=int, default=0, choices=[0, 1, 2, 3], help="Previous owners count (0, 1, 2, 3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Path relative to script location or working directory
    model_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_car_price_model.joblib")
    if not os.path.exists(model_file):
        model_file = "models/best_car_price_model.joblib"
        
    predictor = CarPricePredictor(model_file)
    result = predictor.predict_single(
        car_name=args.car_name,
        year=args.year,
        present_price=args.present_price,
        driven_kms=args.driven_kms,
        fuel_type=args.fuel_type,
        selling_type=args.selling_type,
        transmission=args.transmission,
        owner=args.owner
    )
    
    print("\n" + "="*60)
    print("       VEHICLE RESALE PRICE PREDICTION REPORT")
    print("="*60)
    print(f"Vehicle:                   {result['vehicle_name'].title()} ({result['brand']})")
    print(f"Vehicle Type:              {result['vehicle_type']}")
    print(f"Year of Manufacture:      {result['manufacturing_year']} (Age: {result['vehicle_age_years']} years)")
    print(f"Original Ex-Showroom:      Rs. {result['present_price_lakhs']} Lakhs")
    print(f"Distance Driven:           {args.driven_kms:,} kms")
    print(f"Fuel / Transmission:       {args.fuel_type} / {args.transmission}")
    print(f"Seller Type / Owners:      {args.selling_type} / {args.owner} prior owners")
    print(f"Brand Goodwill Rating:     {result['specifications']['brand_goodwill_score']} / 10.0")
    print(f"Engine Horsepower:         {result['specifications']['horsepower_bhp']} BHP")
    print(f"Fuel Efficiency / Mileage: {result['specifications']['mileage_kmpl']} kmpl")
    print("-" * 60)
    print(f">> PREDICTED RESALE PRICE: Rs. {result['predicted_selling_price_lakhs']} Lakhs ({result['predicted_price_inr']})")
    print(f">> VALUE RETENTION RATIO:  {result['value_retention_percent']}")
    print(f">> ESTIMATED DEPRECIATION: Rs. {result['estimated_depreciation_lakhs']} Lakhs")
    print("="*60 + "\n")
