# TASK 3: Car Price Prediction with Machine Learning

An end-to-end Machine Learning and Data Science workflow for automobile and two-wheeler valuation. This project extracts automotive brand goodwill, enriches vehicles with technical specifications (horsepower, fuel efficiency, engine displacement), derives usage dynamics, benchmarks multiple regression algorithms with 5-fold cross-validation, and provides interactive inference and production-ready pipelines.

---

## 📌 Project Highlights & Achievements
- **Comprehensive Feature Engineering:** Extracted 14 automotive manufacturers, classified passenger cars vs. two-wheelers, formulated a **Brand Goodwill Rating** index (1.0 - 10.0), mapped **Horsepower (BHP)**, **Mileage / Fuel Efficiency (kmpl)**, **Engine Capacity (cc)**, and calculated annual usage intensity (`Km_per_Year`).
- **Multi-Model Benchmark:** Evaluated 7 regression architectures (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and Extra Trees Regressor).
- **Rigorous Cross-Validation:** 5-Fold Cross-Validation on training data to prevent overfitting and guarantee statistical generalization.
- **Top Performing Architecture:** Tuned **Extra Trees Regressor** achieving:
  - **5-Fold Cross-Validation $R^2$:** `0.8938`
  - **Held-Out Test Set $R^2$:** `0.8439` (Adjusted $R^2$: `0.7122`)
  - **Mean Absolute Error (MAE):** `0.8431 Lakhs` (~Rs. 84,310)
  - **Root Mean Squared Error (RMSE):** `2.0059 Lakhs`
  - **Mean Absolute Percentage Error (MAPE):** `14.88%`
- **Interactive CLI & Notebook:** Pre-rendered Jupyter Notebook (`car_price_prediction.ipynb`), modular `src/` codebase, and a dedicated prediction CLI tool (`src/predict.py`).
- **Publication-Quality Visualizations:** 8 high-resolution diagnostic plots generated in `plots/`.

---

## 📂 Repository Structure

```text
data science project 3/
├── car data.csv                         # Raw transactional dataset
├── requirements.txt                     # Python package dependencies
├── run_pipeline.py                      # Master script to execute end-to-end ML pipeline
├── car_price_prediction.ipynb           # Fully executed interactive Jupyter Notebook
├── create_notebook.py                   # Automated notebook build script
├── src/                                 # Modular production codebase
│   ├── __init__.py
│   ├── data_preprocessing.py            # Data quality, validation, duplicate & outlier handling
│   ├── feature_engineering.py           # Brand extraction, goodwill mapping, specs lookup
│   ├── model_training.py                # Multi-model benchmarking, CV, tuning & plotting
│   └── predict.py                       # CLI & Python API for instant vehicle pricing
├── plots/                               # Diagnostic & exploratory visual figures
│   ├── correlation_matrix.png           # Feature correlation heatmap
│   ├── price_distribution.png           # Raw vs. log-transformed target distribution
│   ├── brand_goodwill_analysis.png      # Resale value across brand tiers & goodwill scores
│   ├── depreciation_and_horsepower.png  # Age depreciation curves & horsepower regression
│   ├── model_performance_comparison.png # Benchmark bar charts for Test R² and MAE
│   ├── actual_vs_predicted.png          # Actual vs. predicted 1:1 identity line
│   ├── residual_diagnostics.png         # Residual error distribution & homoscedasticity plot
│   └── feature_importances.png          # Top 15 most influential predictors
└── models/                              # Saved model artifacts & metrics
    ├── best_car_price_model.joblib      # Serialized trained model pipeline
    └── model_metrics.json               # Serialized evaluation metrics & hyperparameters
```

---

## 🛠️ Dataset Description & Preprocessing

The primary dataset (`car data.csv`) comprises secondary automotive market transactions across 301 instances with 9 original features:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Car_Name` | Categorical | Specific model title (e.g., 'swift', 'ciaz', 'fortuner', 'Activa 3g') |
| `Year` | Numerical (Integer) | Year of initial registration (2003 - 2018) |
| `Selling_Price` | Numerical (Float) | Target variable: Resale selling price in Lakhs INR |
| `Present_Price` | Numerical (Float) | Ex-showroom new vehicle price in Lakhs INR |
| `Driven_kms` | Numerical (Integer) | Total distance driven in kilometers |
| `Fuel_Type` | Categorical | Powertrain: 'Petrol', 'Diesel', 'CNG' |
| `Selling_type` | Categorical | Seller channel: 'Dealer' vs. 'Individual' |
| `Transmission` | Categorical | Transmission type: 'Manual' vs. 'Automatic' |
| `Owner` | Numerical (Integer) | Prior owners count (0 = 1st owner, 1 = 2nd owner, 3 = 4th+ owner) |

### Preprocessing Actions:
1. **Missing Value Audit:** Confirmed zero missing/null values across all columns.
2. **Deduplication:** Identified and eliminated 2 duplicate transactions (`ertiga` and `fortuner` duplicate entries) to prevent train-test contamination.
3. **Outlier Diagnosis (IQR Method):** Detected natural upper-tail entries (e.g., `land cruiser` at Rs. 92.6 Lakhs ex-showroom). These were preserved as legitimate luxury vehicle market data and handled robustly through tree ensembles.

---

## ⚙️ Domain-Driven Feature Engineering

Automotive resale value is governed by non-linear physical and commercial variables:

1. **Brand Extraction:** Cleaned string tokens to extract 14 distinct parent manufacturers: *Toyota, Honda, Maruti Suzuki, Hyundai, Royal Enfield, KTM, Yamaha, Bajaj, Hero, TVS, Suzuki, Mahindra, Hyosung, and UM*.
2. **Vehicle Classification:** Automatically differentiated 4-wheel passenger cars (214 rows) from motorcycles/scooters (87 rows).
3. **Brand Goodwill Rating ($1.0 - 10.0$):** Quantified brand trust, parts availability, and historical second-hand value retention in the Indian market:
   - **Toyota (9.5):** Industry-leading value retention (Fortuner & Innova command exceptional resale premiums).
   - **Maruti Suzuki (9.2):** Unmatched nationwide service network, low maintenance, and high second-hand liquidity.
   - **Honda (8.8) & Hyundai (8.6):** High powertrain longevity, modern feature adoption.
   - **Royal Enfield (8.7):** Cult following with minimal depreciation curve.
   - **Mass Market Commuters (7.3 - 7.6):** Steady high-volume resale market.
4. **Vehicle Age & Annual Mileage Intensity:**
   $$\text{Vehicle\_Age} = \text{Reference\_Year (2018)} - \text{Year}$$
   $$\text{Km\_per\_Year} = \frac{\text{Driven\_kms}}{\text{Vehicle\_Age} + 1}$$
5. **Technical Specifications Enrichment:**
   - **`Horsepower_bhp`:** Engine power output (ranging from 8 BHP for scooters to 261 BHP for the Land Cruiser).
   - **`Mileage_kmpl`:** ARAI certified fuel efficiency (ranging from 9.0 kmpl for full-size SUVs to 65.0 kmpl for commuter bikes).
   - **`Engine_Capacity_cc`:** Engine displacement (ranging from 100cc to 4461cc).

---

## 📊 Exploratory Data Analysis & Visual Insights

Key findings from generated exploratory charts:
- **Price Distribution & Skewness:** `Selling_Price` exhibits positive right-skewness with the majority of transactions under Rs. 10 Lakhs. Applying $\log(1 + y)$ normalizes the variance.
- **Strongest Predictors:** `Present_Price` exhibits the highest positive linear correlation ($r = 0.88$) with resale price, followed by `Horsepower_bhp` ($r = 0.80$).
- **Depreciation Trend:** Vehicle price depreciates sharply during the first 3-5 years, followed by an asymptotic flattening in older models.
- **Fuel Type Premium:** Diesel models hold significantly higher nominal resale value than petrol equivalents due to higher commercial utility and torque.

### 📈 Exploratory Visualizations:
<p align="center">
  <img src="plots/correlation_matrix.png" alt="Correlation Matrix" width="48%" />
  <img src="plots/price_distribution.png" alt="Price Distribution" width="48%" />
</p>

<p align="center">
  <img src="plots/brand_goodwill_analysis.png" alt="Brand Goodwill Analysis" width="48%" />
  <img src="plots/depreciation_and_horsepower.png" alt="Depreciation & Horsepower" width="48%" />
</p>

---

## 🏆 Model Benchmarking & Evaluation

Candidate models were trained on 80% of data (239 instances) with 5-Fold Cross-Validation, and evaluated on an independent 20% held-out test set (60 instances):

| Model Architecture | 5-Fold CV $R^2$ | Test Set $R^2$ | Adjusted $R^2$ | Test MAE (Lakhs) | Test RMSE (Lakhs) | Test MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees Regressor (Tuned)** | **0.8938** | **0.8439** | **0.7122** | **0.8431** | **2.0059** | **14.88%** |
| **Decision Tree Regressor** | 0.8778 | 0.7857 | 0.6048 | 1.1232 | 2.3504 | 19.34% |
| **Ridge Regression** | 0.8713 | 0.6935 | 0.4348 | 1.4968 | 2.8107 | 45.12% |
| **Gradient Boosting Regressor** | 0.8935 | 0.6882 | 0.4251 | 1.2016 | 2.8348 | 21.65% |
| **Linear Regression** | 0.8705 | 0.6877 | 0.4243 | 1.5179 | 2.8369 | 46.20% |
| **Lasso Regression ($\alpha=0.05$)** | 0.8740 | 0.6833 | 0.4160 | 1.4377 | 2.8571 | 42.80% |
| **Random Forest Regressor** | 0.9036 | 0.6290 | 0.3159 | 1.2661 | 3.0924 | 22.45% |

### 📊 Model Performance Comparison:
<p align="center">
  <img src="plots/model_performance_comparison.png" alt="Model Benchmark Comparison" width="85%" />
</p>

### 🔬 Diagnostics & Model Interpretability:
<p align="center">
  <img src="plots/actual_vs_predicted.png" alt="Actual vs Predicted" width="48%" />
  <img src="plots/residual_diagnostics.png" alt="Residual Diagnostics" width="48%" />
</p>

<p align="center">
  <img src="plots/feature_importances.png" alt="Feature Importances" width="85%" />
</p>

### Hyperparameter Tuning Results:
Using `GridSearchCV` across candidate parameters for Extra Trees:
```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
```
**Interpretation:** An MAE of **0.84 Lakhs** means that across vehicles ranging from Rs. 10,000 to Rs. 35,00,000, model estimates deviate by only ~Rs. 84,310 on average!

---

## 🚀 How to Run the Project

### 1. Prerequisites & Environment Setup
Clone or navigate to the project directory:
```bash
cd "c:\Users\H-P\Downloads\data science project 3"
```
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Executes data preprocessing, feature engineering, model training, cross-validation, tuning, diagnostic plotting, and test valuations:
```bash
python run_pipeline.py
```

### 3. Interactive CLI Inference Tool
Predict resale prices for any custom vehicle directly from your terminal:
```bash
# Example: Predict resale price for a 2016 Honda City driven 42,000 km
python src/predict.py --car_name "honda city" --year 2016 --present_price 11.5 --driven_kms 42000 --fuel_type "Petrol" --transmission "Manual"

# Example: Predict resale price for a 2017 Toyota Fortuner Diesel
python src/predict.py --car_name "fortuner" --year 2017 --present_price 31.0 --driven_kms 55000 --fuel_type "Diesel" --transmission "Automatic"

# Example: Predict resale price for a Royal Enfield Classic 350
python src/predict.py --car_name "Royal Enfield Classic 350" --year 2016 --present_price 1.5 --driven_kms 18000 --fuel_type "Petrol" --transmission "Manual"
```

### 4. Interactive Jupyter Notebook
Launch the fully executed notebook in VS Code or JupyterLab:
```bash
jupyter notebook car_price_prediction.ipynb
```

---

## 🌐 Real-World Applications of Machine Learning in Price Prediction

Machine learning pricing engines drive core functionality across digital automotive and financial services:

### 1. Instant Algorithmic Buyback Offers (Cars24, Carvana, Spinny, AutoTrader)
- **Zero-Friction Liquidity:** Enables consumers to enter vehicle registration and mileage on mobile apps to receive guaranteed cash quotes in under 60 seconds without manual dealer appraisal.
- **Dynamic Inventory Pricing:** E-commerce platforms continuously adjust listing prices based on inventory aging (days-in-stock), localized search volume, competitor listings, and regional demand shifts.

### 2. Automotive Insurance & Total Loss Adjustments
- **Actual Cash Value (ACV) Calculation:** In major collision or flood claims, insurance adjusters use unbiased regression models to establish the pre-loss fair market value, reducing settlement disputes and eliminating adjuster bias.
- **Diminished Value Claims:** Predicts the post-repair market discount for vehicles with recorded structural or paint repairs.

### 3. Auto Financing & Lease Residual Value Forecasting
- **Lease End Residual Value Estimation:** Monthly lease payments are calculated as the difference between the initial capitalized cost and the projected *residual value* at lease termination (36-48 months later). Machine learning residual models protect financial institutions against multi-million-dollar portfolio depreciation write-downs.
- **Loan-to-Value (LTV) Risk Guardrails:** Safeguards banks against negative equity loans by modeling vehicle depreciation trajectories relative to loan amortization curves.

### 4. Fleet Management & Commercial Liquidation Optimization
- **Optimal Replacement Timing:** Fleet operators (rental companies, delivery logistics, corporate fleets) use cost-curve modeling to pinpoint the inflection point where rising maintenance and downtime costs exceed depreciation losses, indicating the exact month to auction the asset.

### 5. Regulatory Audits & Taxation Compliance
- Tax authorities and customs agencies use standardized ML valuation models to audit second-hand vehicle transfers, preventing invoice under-reporting and tax evasion.

---

## 📜 Conclusion
This project demonstrates that combining **domain-specific feature engineering** (brand goodwill index, horsepower, fuel efficiency, usage intensity) with **non-parametric ensemble regressors** (Extra Trees) yields an accurate ($R^2 = 0.8439$, $\text{MAE} \approx \text{Rs. } 84,310$), interpretable, and production-ready automotive pricing engine.
