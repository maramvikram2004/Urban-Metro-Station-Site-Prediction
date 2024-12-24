#xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# Load datasets
df_bangalore = pd.read_csv('/content/datasset1.csv')
df_hyderabad = pd.read_csv('/content/hydexcel.csv')

# Clean column names and target variable
df_bangalore.columns = df_bangalore.columns.str.strip()
df_hyderabad.columns = df_hyderabad.columns.str.strip()
df_bangalore['metro'] = df_bangalore['metro'].map({'y': 1, 'n': 0})
df_hyderabad['metro'] = df_hyderabad['metro'].map({'y': 1, 'n': 0})

# List of numeric columns to clean
numeric_columns = ['popden', 'hospitals', 'entertainment', 'monuments', 'schools', 'rent',
                   'lat', 'long', 'railway', 'traffic ']

# Convert numeric columns to float after removing commas
for col in numeric_columns:
    if col in df_bangalore.columns:
        df_bangalore[col] = df_bangalore[col].astype(str).str.replace(',', '').astype(float)
    if col in df_hyderabad.columns:
        df_hyderabad[col] = df_hyderabad[col].astype(str).str.replace(',', '').astype(float)

# Feature engineering: Add 'buildings' as a derived feature
df_bangalore['buildings'] = df_bangalore[['hospitals', 'entertainment', 'monuments']].sum(axis=1)
df_hyderabad['buildings'] = df_hyderabad[['hospitals', 'entertainment', 'monuments']].sum(axis=1)

# Define features and target variable
selected_features = ['popden', 'buildings', 'schools', 'rent', 'lat', 'long', 'railway', 'traffic']
target_variable = 'metro'

# Verify selected features exist in datasets
missing_features_bangalore = set(selected_features) - set(df_bangalore.columns)
missing_features_hyderabad = set(selected_features) - set(df_hyderabad.columns)
if missing_features_bangalore or missing_features_hyderabad:
    raise ValueError(f"Missing features. Bangalore: {missing_features_bangalore}, Hyderabad: {missing_features_hyderabad}")

# Normalize datasets
df_bangalore_mean = df_bangalore[selected_features].mean()
df_bangalore[selected_features] /= df_bangalore_mean

df_hyderabad_mean = df_hyderabad[selected_features].mean()
df_hyderabad[selected_features] /= df_hyderabad_mean

# Combine datasets
df_combined = pd.concat([df_bangalore, df_hyderabad], ignore_index=True)

# Split data into features and target
X_combined = df_combined[selected_features]
y_combined = df_combined[target_variable]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Handle class imbalance using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Define and train the XGBoost classifier
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_test_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, pos_label=1)

print(f"Testing Accuracy: {accuracy}")
print(f"Testing F1 Score: {f1}")

# Predict for new samples
new_samples = pd.read_csv('/content/chennaiexcel.csv')
new_samples.columns = new_samples.columns.str.strip()

# Clean and normalize new samples
for col in numeric_columns:
    if col in new_samples.columns:
        new_samples[col] = new_samples[col].astype(str).str.replace(',', '').astype(float)
new_samples[selected_features] /= new_samples[selected_features].mean()

# Scale latitude and longitude
scaler = MinMaxScaler()
new_samples[['lat', 'long']] = scaler.fit_transform(new_samples[['lat', 'long']])

# Predict metro installations
predictions = xgb_model.predict(new_samples[selected_features])
new_samples['prediction'] = predictions

# Filter predicted metro locations
predicted_metro_locations = new_samples[new_samples['prediction'] == 1]

# Plot predictions
fig, ax = plt.subplots(figsize=(10, 8))
map_image = plt.imread('/content/chennai_final.png')
ax.imshow(map_image, extent=[0, 1, 0, 1])

# Plot metro predictions
ax.scatter(new_samples['long'], new_samples['lat'], c=['red' if pred == 1 else 'black' for pred in predictions], marker='x', s=100)

# Draw connections between predicted metro locations
for i in range(len(predicted_metro_locations) - 1):
    ax.plot([predicted_metro_locations['long'].iloc[i], predicted_metro_locations['long'].iloc[i + 1]],
            [predicted_metro_locations['lat'].iloc[i], predicted_metro_locations['lat'].iloc[i + 1]], color='red')

plt.title('Metro Installation Predictions')
plt.xlabel('Normalized Longitude')
plt.ylabel('Normalized Latitude')
plt.show()
