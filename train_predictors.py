"""
Train ML Prediction Models for Healthcare Resource Allocation
--------------------------------------------------------------
Model 1: Budget Predictor (Gradient Boosting Regressor)
Model 2: Resource Predictor (Multi-output Random Forest Regressor)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Healthcare Prediction Model Training")
print("=" * 60)

# -------------------------
# Load Data
# -------------------------
print("\n[1/6] Loading healthcare data...")
df = pd.read_csv("data/india_healthcare_data.csv")
print(f"   Loaded {len(df)} records across {df['state'].nunique()} states/UTs")
print(f"   Years: {df['year'].min()} to {df['year'].max()}")

# -------------------------
# Feature Engineering
# -------------------------
print("\n[2/6] Engineering features...")

# Encode states
le_state = LabelEncoder()
df['state_encoded'] = le_state.fit_transform(df['state'])

# Save state encoder for inference
joblib.dump(le_state, "state_encoder.pkl")
print(f"   State encoder saved ({len(le_state.classes_)} states)")

# Calculate derived features
df['population_millions'] = df['population_crore'] * 10
df['urban_rural_gap'] = abs(df['urban_pct'] - df['rural_pct'])
df['health_workforce_density'] = (df['doctors_total'] + df['nurses_total']) / df['population_millions']
df['infrastructure_composite'] = (
    df['hospital_beds_per_1000'] * 0.3 +
    df['doctor_per_1000'] * 0.3 +
    (df['vaccine_coverage_pct'] / 100) * 0.2 +
    (1 - df['infra_gap_score'] / 10) * 0.2
)

# -------------------------
# Model 1: Budget Predictor
# -------------------------
print("\n[3/6] Training Budget Predictor (Gradient Boosting)...")

budget_features = [
    'state_encoded', 'year', 'population_crore', 'disease_index',
    'infra_gap_score', 'urban_pct', 'hospitals_total', 'doctors_total',
    'vaccine_coverage_pct', 'maternal_mortality_ratio', 'infant_mortality_rate'
]

X_budget = df[budget_features].values
y_budget = df['health_budget_crore'].values

budget_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)

budget_model.fit(X_budget, y_budget)

# Cross-validation
budget_scores = cross_val_score(budget_model, X_budget, y_budget, cv=5, scoring='r2')
print(f"   R² Score (CV): {budget_scores.mean():.4f} ± {budget_scores.std():.4f}")

# Feature importance
budget_importance = sorted(
    zip(budget_features, budget_model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
print("   Top features:")
for feat, imp in budget_importance[:5]:
    print(f"      {feat}: {imp:.3f}")

joblib.dump(budget_model, "budget_predictor.pkl")
print("   Budget model saved as budget_predictor.pkl")

# Save feature list for inference
joblib.dump(budget_features, "budget_features.pkl")

# -------------------------
# Model 2: Resource Predictor
# -------------------------
print("\n[4/6] Training Resource Predictor (Multi-output Random Forest)...")

resource_features = [
    'state_encoded', 'year', 'population_crore', 'disease_index',
    'infra_gap_score', 'urban_pct', 'health_budget_crore'
]

resource_targets = [
    'hospitals_total', 'doctors_total', 'vaccine_doses_required_cr',
    'icu_beds', 'hospital_beds_per_1000', 'nurses_total'
]

X_resource = df[resource_features].values
y_resource = df[resource_targets].values

resource_model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
)

resource_model.fit(X_resource, y_resource)

# Per-target R² scores
print("   Per-target R² scores:")
for i, target in enumerate(resource_targets):
    y_pred_single = resource_model.predict(X_resource)[:, i]
    from sklearn.metrics import r2_score
    r2 = r2_score(y_resource[:, i], y_pred_single)
    print(f"      {target}: {r2:.4f}")

joblib.dump(resource_model, "resource_predictor.pkl")
print("   Resource model saved as resource_predictor.pkl")

# Save feature and target lists
joblib.dump(resource_features, "resource_features.pkl")
joblib.dump(resource_targets, "resource_targets.pkl")

# -------------------------
# Create 2028-29 Predictions
# -------------------------
print("\n[5/6] Generating 2028-29 predictions...")

predictions = []

for state in df['state'].unique():
    state_data = df[df['state'] == state].sort_values('year')
    latest = state_data.iloc[-1]  # 2027 data

    for target_year in [2028, 2029]:
        # Project forward using trends
        growth_rate_pop = (state_data['population_crore'].iloc[-1] / state_data['population_crore'].iloc[0]) ** (1/7) - 1
        growth_rate_urban = (state_data['urban_pct'].iloc[-1] - state_data['urban_pct'].iloc[0]) / 7

        proj_population = latest['population_crore'] * (1 + growth_rate_pop) ** (target_year - 2027)
        proj_urban = min(latest['urban_pct'] + growth_rate_urban * (target_year - 2027), 99)
        proj_disease = max(latest['disease_index'] - 0.01 * (target_year - 2027), 0.15)
        proj_infra_gap = max(latest['infra_gap_score'] - 0.2 * (target_year - 2027), 1.0)
        proj_hospitals = latest['hospitals_total'] * (1 + 0.035)  ** (target_year - 2027)
        proj_doctors = latest['doctors_total'] * (1 + 0.03) ** (target_year - 2027)
        proj_vaccine_cov = min(latest['vaccine_coverage_pct'] + 1.5 * (target_year - 2027), 98)
        proj_mmr = max(latest['maternal_mortality_ratio'] - 3 * (target_year - 2027), 15)
        proj_imr = max(latest['infant_mortality_rate'] - 1.5 * (target_year - 2027), 4)

        state_enc = le_state.transform([state])[0]

        # Budget prediction
        budget_input = np.array([[
            state_enc, target_year, proj_population, proj_disease,
            proj_infra_gap, proj_urban, proj_hospitals, proj_doctors,
            proj_vaccine_cov, proj_mmr, proj_imr
        ]])
        predicted_budget = budget_model.predict(budget_input)[0]

        # Resource prediction
        resource_input = np.array([[
            state_enc, target_year, proj_population, proj_disease,
            proj_infra_gap, proj_urban, predicted_budget
        ]])
        predicted_resources = resource_model.predict(resource_input)[0]

        predictions.append({
            'state': str(state),
            'year': int(target_year),
            'population_crore': float(round(proj_population, 2)),
            'disease_index': float(round(proj_disease, 2)),
            'infra_gap_score': float(round(proj_infra_gap, 1)),
            'urban_pct': float(round(proj_urban, 1)),
            'predicted_budget_crore': float(round(predicted_budget, 0)),
            'predicted_hospitals': float(round(predicted_resources[0], 0)),
            'predicted_doctors': float(round(predicted_resources[1], 0)),
            'predicted_vaccine_doses_cr': float(round(predicted_resources[2], 2)),
            'predicted_icu_beds': float(round(predicted_resources[3], 0)),
            'predicted_beds_per_1000': float(round(predicted_resources[4], 2)),
            'predicted_nurses': float(round(predicted_resources[5], 0)),
            'projected_vaccine_coverage_pct': float(round(proj_vaccine_cov, 1)),
            'projected_mmr': float(round(proj_mmr, 0)),
            'projected_imr': float(round(proj_imr, 0))
        })

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("data/predictions_2028_29.csv", index=False)
print(f"   Saved {len(predictions_df)} predictions to data/predictions_2028_29.csv")

# Also save as JSON for easy loading
with open("data/predictions_2028_29.json", "w") as f:
    json.dump(predictions, f, indent=2)

# -------------------------
# Summary Statistics
# -------------------------
print("\n[6/6] Summary")
print("=" * 60)
print(f"   Training data: {len(df)} records")
print(f"   States/UTs covered: {df['state'].nunique()}")
print(f"   Budget model R²: {budget_scores.mean():.4f}")
print(f"   Predictions generated: {len(predictions)} (2028 + 2029)")
print(f"\n   Files created:")
print(f"      budget_predictor.pkl")
print(f"      resource_predictor.pkl")
print(f"      state_encoder.pkl")
print(f"      budget_features.pkl")
print(f"      resource_features.pkl")
print(f"      resource_targets.pkl")
print(f"      data/predictions_2028_29.csv")
print(f"      data/predictions_2028_29.json")
print("=" * 60)
print("\nAll models trained successfully!")
