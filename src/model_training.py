"""
Model Training and Evaluation Module for Car Price Prediction
Includes multi-model benchmarking, 5-Fold cross-validation, hyperparameter tuning,
diagnostic visualization generation, and model serialization.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib


def calculate_adjusted_r2(r2, n, p):
    """Calculate Adjusted R-squared given sample size n and predictor count p."""
    if n - p - 1 <= 0:
        return r2
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


def prepare_features_and_target(df_eng):
    """
    Encode categorical features and split into predictors (X) and target (y).
    """
    cat_cols = ['Fuel_Type', 'Selling_type', 'Transmission', 'Vehicle_Type', 'Brand']
    
    # Columns to exclude from feature matrix
    ignore_cols = ['Car_Name', 'Selling_Price', 'Year', 'Price_Depreciation', 'Value_Retention_Ratio']
    feature_source = df_eng.drop(columns=[col for col in ignore_cols if col in df_eng.columns])
    
    # One-hot encode categoricals
    X_encoded = pd.get_dummies(feature_source, columns=cat_cols, drop_first=True)
    y = df_eng['Selling_Price']
    
    return X_encoded, y


def generate_eda_plots(df_eng, output_dir="plots"):
    """
    Generate and save comprehensive EDA and exploratory visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.1)
    
    # 1. Price Distribution & Log Price
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.distplot(df_eng['Selling_Price'], kde=True, color='teal', ax=axes[0], bins=25)
    axes[0].set_title("Selling Price Distribution (in Lakhs)", fontweight='bold')
    axes[0].set_xlabel("Selling Price (Lakhs INR)")
    axes[0].set_ylabel("Density")
    
    sns.distplot(np.log1p(df_eng['Selling_Price']), kde=True, color='coral', ax=axes[1], bins=25)
    axes[1].set_title("Log-Transformed Selling Price Distribution", fontweight='bold')
    axes[1].set_xlabel("log(1 + Selling Price)")
    axes[1].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "price_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Correlation Matrix Heatmap
    num_cols = ['Selling_Price', 'Present_Price', 'Driven_kms', 'Vehicle_Age', 
                'Brand_Goodwill', 'Horsepower_bhp', 'Mileage_kmpl', 'Km_per_Year']
    valid_num_cols = [c for c in num_cols if c in df_eng.columns]
    
    plt.figure(figsize=(10, 8))
    corr = df_eng[valid_num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
                mask=mask, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap: Automotive Features vs. Selling Price", fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    
    # 3. Brand Goodwill & Value Retention Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    brand_order = df_eng.groupby('Brand')['Selling_Price'].median().sort_values(ascending=False).index
    sns.boxplot(x='Brand', y='Selling_Price', data=df_eng, order=brand_order, palette='viridis', ax=axes[0])
    axes[0].set_title("Selling Price Distribution by Automotive Brand", fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].set_ylabel("Selling Price (Lakhs)")
    
    sns.scatterplot(x='Brand_Goodwill', y='Selling_Price', hue='Vehicle_Type', 
                    size='Present_Price', sizes=(30, 250), alpha=0.8, data=df_eng, palette='Set1', ax=axes[1])
    axes[1].set_title("Selling Price vs. Brand Goodwill Rating", fontweight='bold')
    axes[1].set_xlabel("Brand Goodwill Rating (1-10 Scale)")
    axes[1].set_ylabel("Selling Price (Lakhs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "brand_goodwill_analysis.png"), dpi=300)
    plt.close()
    
    # 4. Age and Horsepower vs Selling Price
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.regplot(x='Vehicle_Age', y='Selling_Price', data=df_eng, 
                scatter_kws={'alpha': 0.6, 'color': 'navy'}, line_kws={'color': 'red', 'linewidth': 2}, ax=axes[0])
    axes[0].set_title("Impact of Vehicle Age on Resale Value (Depreciation)", fontweight='bold')
    axes[0].set_xlabel("Vehicle Age (Years)")
    axes[0].set_ylabel("Selling Price (Lakhs)")
    
    sns.regplot(x='Horsepower_bhp', y='Selling_Price', data=df_eng,
                scatter_kws={'alpha': 0.6, 'color': 'darkgreen'}, line_kws={'color': 'darkorange', 'linewidth': 2}, ax=axes[1])
    axes[1].set_title("Engine Horsepower (BHP) vs. Selling Price", fontweight='bold')
    axes[1].set_xlabel("Horsepower (BHP)")
    axes[1].set_ylabel("Selling Price (Lakhs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "depreciation_and_horsepower.png"), dpi=300)
    plt.close()
    
    print(f"[INFO] Exploratory visual plots saved to '{output_dir}/'.")


def benchmark_models(X_train, X_test, y_train, y_test, n_splits=5):
    """
    Train and evaluate multiple regression algorithms with cross-validation and test set metrics.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.05),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42)
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    trained_models = {}
    
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'CV R2 (5-Fold)':<15} | {'Test R2':<10} | {'Adj R2':<10} | {'MAE':<8} | {'RMSE':<8}")
    print("="*80)
    
    for name, model in models.items():
        # Cross validation scores on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_r2_mean = float(np.mean(cv_scores))
        cv_r2_std = float(np.std(cv_scores))
        
        # Fit on entire training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Metrics
        test_r2 = float(r2_score(y_test, test_preds))
        adj_r2 = float(calculate_adjusted_r2(test_r2, len(y_test), X_test.shape[1]))
        mae = float(mean_absolute_error(y_test, test_preds))
        mse = float(mean_squared_error(y_test, test_preds))
        rmse = float(np.sqrt(mse))
        mape = float(mean_absolute_percentage_error(y_test, test_preds))
        
        results.append({
            'Model': name,
            'CV_R2_Mean': cv_r2_mean,
            'CV_R2_Std': cv_r2_std,
            'Train_R2': float(r2_score(y_train, train_preds)),
            'Test_R2': test_r2,
            'Adjusted_R2': adj_r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        })
        
        cv_display = f"{cv_r2_mean:.4f} +/- {cv_r2_std:.3f}"
        print(f"{name:<20} | {cv_display:<15} | {test_r2:<10.4f} | {adj_r2:<10.4f} | {mae:<8.4f} | {rmse:<8.4f}")
        
    print("="*80)
    return pd.DataFrame(results), trained_models


def tune_best_model(X_train, y_train):
    """
    Perform Hyperparameter optimization using GridSearchCV for the top performing ExtraTrees Regressor.
    """
    print("\n[INFO] Starting Hyperparameter Optimization for Extra Trees Regressor...")
    param_grid = {
        'n_estimators': [75, 100, 125],
        'max_depth': [8, 10, 12, None],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2]
    }
    
    base_model = ExtraTreesRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"[INFO] Best Hyperparameters found: {grid_search.best_params_}")
    print(f"[INFO] Best Cross-Validation R2: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_


def plot_model_evaluation(best_model, X_test, y_test, results_df, feature_names, output_dir="plots"):
    """
    Generate diagnostic evaluation plots:
    1. Benchmark comparison bar chart
    2. Actual vs. Predicted values with 45-degree reference line
    3. Residual distribution and Q-Q diagnostics
    4. Top 15 Feature Importances
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.1)
    
    # 1. Model Benchmark Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(x='Test_R2', y='Model', data=results_df.sort_values('Test_R2', ascending=False), 
                palette='Blues_r', ax=axes[0])
    axes[0].set_title("Test R² Score Comparison Across Models", fontweight='bold')
    axes[0].set_xlabel("R² Score (Higher is better)")
    axes[0].set_xlim(0.5, 1.0)
    
    sns.barplot(x='MAE', y='Model', data=results_df.sort_values('MAE'), 
                palette='Greens_r', ax=axes[1])
    axes[1].set_title("Mean Absolute Error (MAE) Comparison (Lakhs)", fontweight='bold')
    axes[1].set_xlabel("MAE in Lakhs INR (Lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_performance_comparison.png"), dpi=300)
    plt.close()
    
    # Best Model Predictions
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred
    
    # 2. Actual vs Predicted
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_pred, alpha=0.75, color='royalblue', edgecolors='k', s=60)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Ideal 1:1 Identity Line')
    plt.xlabel("Actual Selling Price (Lakhs INR)", fontweight='bold')
    plt.ylabel("Predicted Selling Price (Lakhs INR)", fontweight='bold')
    plt.title(f"Actual vs. Predicted Price (Test R² = {r2_score(y_test, y_pred):.4f})", fontweight='bold', fontsize=13)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    # 3. Residual Diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.distplot(residuals, kde=True, color='purple', ax=axes[0], bins=20)
    axes[0].axvline(0, color='red', linestyle='--', lw=2)
    axes[0].set_title("Residuals Distribution (Actual - Predicted)", fontweight='bold')
    axes[0].set_xlabel("Residual Error (Lakhs)")
    axes[0].set_ylabel("Density")
    
    axes[1].scatter(y_pred, residuals, alpha=0.75, color='darkorange', edgecolors='k')
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_title("Residuals vs. Predicted Values (Homoscedasticity Check)", fontweight='bold')
    axes[1].set_xlabel("Predicted Selling Price (Lakhs)")
    axes[1].set_ylabel("Residual Error (Lakhs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_diagnostics.png"), dpi=300)
    plt.close()
    
    # 4. Feature Importances
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
        top15 = importances.sort_values(ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top15.values, y=top15.index, palette='mako')
        plt.title("Top 15 Most Influential Features (Tree Importance)", fontweight='bold', fontsize=13)
        plt.xlabel("Relative Feature Importance")
        plt.ylabel("Engineered Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=300)
        plt.close()
        
    print(f"[INFO] Evaluation plots saved to '{output_dir}/'.")


def save_pipeline_artifacts(model, feature_columns, metrics, output_dir="models"):
    """
    Save the serialized trained model, feature metadata, and evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "best_car_price_model.joblib")
    joblib.dump({
        'model': model,
        'feature_columns': feature_columns
    }, model_path)
    
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[INFO] Saved serialized model to '{model_path}'.")
    print(f"[INFO] Saved metrics summary to '{metrics_path}'.")


def run_training_pipeline(csv_path="car data.csv", output_dir="models", plots_dir="plots"):
    """
    Execute full training pipeline.
    """
    from data_preprocessing import load_dataset, clean_dataset, inspect_data_quality
    from feature_engineering import engineer_car_features
    
    # 1. Load and clean
    raw_df = load_dataset(csv_path)
    inspect_data_quality(raw_df)
    clean_df = clean_dataset(raw_df)
    
    # 2. Engineer features
    df_eng = engineer_car_features(clean_df)
    
    # 3. Generate EDA plots
    generate_eda_plots(df_eng, output_dir=plots_dir)
    
    # 4. Prepare features and target
    X, y = prepare_features_and_target(df_eng)
    feature_names = list(X.columns)
    
    # 5. Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n[INFO] Data partitioned: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows, Predictors={X_train.shape[1]}")
    
    # 6. Benchmark models
    results_df, trained_models = benchmark_models(X_train, X_test, y_train, y_test)
    
    # 7. Hyperparameter tuning on best candidate (Extra Trees)
    tuned_model, best_params = tune_best_model(X_train, y_train)
    
    # Evaluate tuned model
    tuned_preds = tuned_model.predict(X_test)
    tuned_metrics = {
        'Test_R2': float(r2_score(y_test, tuned_preds)),
        'Adjusted_R2': float(calculate_adjusted_r2(r2_score(y_test, tuned_preds), len(y_test), X_test.shape[1])),
        'MAE': float(mean_absolute_error(y_test, tuned_preds)),
        'MSE': float(mean_squared_error(y_test, tuned_preds)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, tuned_preds))),
        'MAPE': float(mean_absolute_percentage_error(y_test, tuned_preds)),
        'Best_Hyperparameters': best_params
    }
    
    print("\n--- Final Tuned Model Evaluation on Unseen Test Set ---")
    print(f"R² Score:        {tuned_metrics['Test_R2']:.4f}")
    print(f"Adjusted R²:     {tuned_metrics['Adjusted_R2']:.4f}")
    print(f"MAE (Lakhs):     {tuned_metrics['MAE']:.4f}  (~Rs. {int(tuned_metrics['MAE']*100000):,})")
    print(f"RMSE (Lakhs):    {tuned_metrics['RMSE']:.4f} (~Rs. {int(tuned_metrics['RMSE']*100000):,})")
    print(f"MAPE:            {tuned_metrics['MAPE']*100:.2f}%")
    
    # 8. Plot evaluation visuals
    plot_model_evaluation(tuned_model, X_test, y_test, results_df, feature_names, output_dir=plots_dir)
    
    # 9. Save artifacts
    save_pipeline_artifacts(tuned_model, feature_names, tuned_metrics, output_dir=output_dir)
    
    return tuned_model, feature_names, tuned_metrics, results_df


if __name__ == "__main__":
    run_training_pipeline()
