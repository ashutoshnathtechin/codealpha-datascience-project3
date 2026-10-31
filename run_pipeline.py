"""
End-to-End Pipeline Execution Script
Runs complete Machine Learning workflow for Car Price Prediction:
1. Data Ingestion & Quality Inspection
2. Feature Engineering & Brand Goodwill Mapping
3. Exploratory Data Visualization
4. Multi-Model Regression Benchmarking & 5-Fold Cross Validation
5. Hyperparameter Optimization (GridSearchCV)
6. Residual Diagnostics & Feature Importance Analysis
7. Pipeline Model Serialization
8. Demonstration of Real-World Price Predictions
"""

import os
import sys

# Ensure src directory is on Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_dataset, inspect_data_quality, clean_dataset, detect_outliers_iqr
from feature_engineering import engineer_car_features
from model_training import (
    generate_eda_plots,
    prepare_features_and_target,
    benchmark_models,
    tune_best_model,
    plot_model_evaluation,
    save_pipeline_artifacts,
    calculate_adjusted_r2
)
from predict import CarPricePredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np


def main():
    print("=" * 80)
    print("      AUTOMOTIVE PRICE PREDICTION MACHINE LEARNING PIPELINE")
    print("=" * 80)
    
    # 1. Ingestion & Quality
    dataset_path = "car data.csv"
    raw_df = load_dataset(dataset_path)
    quality_summary = inspect_data_quality(raw_df)
    clean_df = clean_dataset(raw_df, drop_duplicates=True)
    outlier_summary = detect_outliers_iqr(clean_df)
    
    # 2. Feature Engineering
    print("\n[STEP 2] Performing Feature Engineering & Specifications Enrichment...")
    df_eng = engineer_car_features(clean_df, reference_year=2018)
    print(f"Features created successfully. Columns: {list(df_eng.columns)}")
    
    # 3. Exploratory Data Analysis & Plots
    print("\n[STEP 3] Generating Exploratory & Domain Visualizations...")
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    generate_eda_plots(df_eng, output_dir="plots")
    
    # 4. Feature Matrix Preparation
    print("\n[STEP 4] Preparing Feature Matrix & Target...")
    X, y = prepare_features_and_target(df_eng)
    feature_names = list(X.columns)
    
    # Train / Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training instances: {X_train.shape[0]} | Testing instances: {X_test.shape[0]} | Features: {X_train.shape[1]}")
    
    # 5. Multi-Model Benchmark
    print("\n[STEP 5] Benchmarking Multiple Regression Models with 5-Fold Cross Validation...")
    benchmark_df, trained_models = benchmark_models(X_train, X_test, y_train, y_test, n_splits=5)
    
    # 6. Hyperparameter Optimization
    print("\n[STEP 6] Tuning Best Performing Architecture...")
    tuned_model, best_params = tune_best_model(X_train, y_train)
    
    # Test evaluation
    y_test_pred = tuned_model.predict(X_test)
    test_r2 = float(r2_score(y_test, y_test_pred))
    adj_r2 = float(calculate_adjusted_r2(test_r2, len(y_test), X_test.shape[1]))
    mae = float(mean_absolute_error(y_test, y_test_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    mape = float(mean_absolute_percentage_error(y_test, y_test_pred))
    
    final_metrics = {
        'Test_R2': round(test_r2, 4),
        'Adjusted_R2': round(adj_r2, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE_pct': round(mape * 100, 2),
        'Best_Hyperparameters': best_params
    }
    
    # 7. Diagnostic Plots & Artifacts
    print("\n[STEP 7] Generating Diagnostic Plots & Saving Artifacts...")
    plot_model_evaluation(tuned_model, X_test, y_test, benchmark_df, feature_names, output_dir="plots")
    save_pipeline_artifacts(tuned_model, feature_names, final_metrics, output_dir="models")
    
    # 8. Real-World Prediction Test Cases
    print("\n[STEP 8] Demonstrating Real-World Vehicle Valuation Examples...")
    predictor = CarPricePredictor("models/best_car_price_model.joblib")
    
    test_cases = [
        {"car_name": "swift", "year": 2015, "present_price": 6.8, "driven_kms": 38000, "fuel_type": "Petrol", "transmission": "Manual"},
        {"car_name": "city", "year": 2016, "present_price": 12.0, "driven_kms": 42000, "fuel_type": "Petrol", "transmission": "Automatic"},
        {"car_name": "fortuner", "year": 2017, "present_price": 31.0, "driven_kms": 55000, "fuel_type": "Diesel", "transmission": "Automatic"},
        {"car_name": "Royal Enfield Classic 350", "year": 2016, "present_price": 1.5, "driven_kms": 18000, "fuel_type": "Petrol", "transmission": "Manual"}
    ]
    
    print("-" * 80)
    print(f"{'Vehicle':<28} | {'Year':<6} | {'Original':<10} | {'Pred Resale':<14} | {'Retention'}")
    print("-" * 80)
    for tc in test_cases:
        res = predictor.predict_single(
            car_name=tc["car_name"],
            year=tc["year"],
            present_price=tc["present_price"],
            driven_kms=tc["driven_kms"],
            fuel_type=tc["fuel_type"],
            transmission=tc["transmission"]
        )
        orig_str = f"Rs. {res['present_price_lakhs']}L"
        pred_str = f"Rs. {res['predicted_selling_price_lakhs']}L ({res['predicted_price_inr']})"
        print(f"{res['vehicle_name'].title():<28} | {res['manufacturing_year']:<6} | {orig_str:<10} | {pred_str:<14} | {res['value_retention_percent']}")
    print("-" * 80)
    
    print("\n[SUCCESS] Pipeline executed successfully! All models, metrics, and plots generated.")


if __name__ == "__main__":
    main()
