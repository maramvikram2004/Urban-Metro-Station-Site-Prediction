#random forest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Load data
df_bangalore = pd.read_csv('/content/datasset1.csv')
df_hyderabad = pd.read_csv('/content/hydexcel.csv')

# Strip any leading/trailing spaces from column names
df_bangalore.columns = df_bangalore.columns.str.strip()
df_hyderabad.columns = df_hyderabad.columns.str.strip()

# Convert 'metro' column to numeric
df_bangalore['metro'] = df_bangalore['metro'].map({'y': 1, 'n': 0})
df_hyderabad['metro'] = df_hyderabad['metro'].map({'y': 1, 'n': 0})

# List of numeric columns to clean
numeric_columns = ['popden', 'hospitals', 'entertainment', 'monuments', 'schools', 'rent', 'lat', 'long', 'railway', 'traffic']

# Remove commas and convert to numeric for relevant columns
for col in numeric_columns:
    if col in df_bangalore.columns:
        df_bangalore[col] = df_bangalore[col].astype(str).str.replace(',', '').astype(float)
    if col in df_hyderabad.columns:
        df_hyderabad[col] = df_hyderabad[col].astype(str).str.replace(',', '').astype(float)

# Feature engineering
df_bangalore['buildings'] = df_bangalore[['hospitals', 'entertainment', 'monuments']].sum(axis=1)
df_hyderabad['buildings'] = df_hyderabad[['hospitals', 'entertainment', 'monuments']].sum(axis=1)

selected_features = ['popden', 'buildings', 'schools', 'rent', 'lat', 'long', 'railway', 'traffic']
target_variable = 'metro'

# Ensure all selected features are present
missing_features_bangalore = set(selected_features) - set(df_bangalore.columns)
missing_features_hyderabad = set(selected_features) - set(df_hyderabad.columns)

if missing_features_bangalore or missing_features_hyderabad:
    raise ValueError(f"Missing features in the datasets. Bangalore: {missing_features_bangalore}, Hyderabad: {missing_features_hyderabad}")

# Normalize data
df_bangalore_mean = df_bangalore[selected_features].mean()
df_bangalore[selected_features] = df_bangalore[selected_features] / df_bangalore_mean

df_hyderabad_mean = df_hyderabad[selected_features].mean()
df_hyderabad[selected_features] = df_hyderabad[selected_features] / df_hyderabad_mean

# Combine dataframes
df_combined = pd.concat([df_bangalore, df_hyderabad], ignore_index=True)

# Split data into features and target
X_combined = df_combined[selected_features]
y_combined = df_combined[target_variable]

# Train-test split
X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Handle class imbalance
oversampler = RandomOverSampler(random_state=42)
X_combined_resampled, y_combined_resampled = oversampler.fit_resample(X_combined_train, y_combined_train)

# Hyperparameter tuning with RandomizedSearchCV for Random Forest
param_distributions_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)
random_search_rf = RandomizedSearchCV(rf_classifier, param_distributions_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search_rf.fit(X_combined_resampled, y_combined_resampled)

# Best model
best_rf_classifier = random_search_rf.best_estimator_

# Predictions and evaluation
y_combined_test_pred_rf = best_rf_classifier.predict(X_combined_test)
test_accuracy_combined_rf = accuracy_score(y_combined_test, y_combined_test_pred_rf)
test_f1_combined_rf = f1_score(y_combined_test, y_combined_test_pred_rf, pos_label=1)

print("Combined Testing F1 Score (Random Forest):", test_f1_combined_rf)

# New samples prediction
new_samples = pd.read_csv('/content/chennaiexcel.csv')

# Strip any leading/trailing spaces from column names
new_samples.columns = new_samples.columns.str.strip()

# Remove commas and convert to numeric for new samples
for col in numeric_columns:
    if col in new_samples.columns:
        new_samples[col] = new_samples[col].astype(str).str.replace(',', '').astype(float)

# Normalize new samples
new_samples[selected_features] = new_samples[selected_features] / new_samples[selected_features].mean()

scaler = MinMaxScaler()
new_samples[['lat', 'long']] = scaler.fit_transform(new_samples[['lat', 'long']])

# Ensure columns match exactly
new_samples_features = [col for col in selected_features if col in new_samples.columns]
predictions_rf = best_rf_classifier.predict(new_samples[new_samples_features])

red_crosses_rf = new_samples[predictions_rf == 1].sort_values(by=['lat', 'long'])

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

map_image = plt.imread('/content/chennai_final.png')

ax.imshow(map_image, extent=[0, 1, 0, 1])

ax.scatter(new_samples['long'], new_samples['lat'], c=['red' if pred == 1 else 'black' for pred in predictions_rf], marker='x', s=100)

for i in range(len(red_crosses_rf) - 1):
    ax.plot([red_crosses_rf['long'].iloc[i], red_crosses_rf['long'].iloc[i + 1]],
            [red_crosses_rf['lat'].iloc[i], red_crosses_rf['lat'].iloc[i + 1]], color='red')

plt.title('Metro Installation Prediction for New Samples (Random Forest)')
plt.xlabel('Normalized Longitude')
plt.ylabel('Normalized Latitude')

plt.show()
