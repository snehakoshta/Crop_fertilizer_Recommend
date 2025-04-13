
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import pickle

# # Load Crop Recommendation Data
# crop_data = pd.read_csv("E:/crop1/Crop_recommendation.csv")
# print("Columns in Crop Dataset:", crop_data.columns)

# # Load Fertilizer Recommendation Data
# fertilizer_data = pd.read_csv("E:/crop1/Fertilizer Prediction.csv")
# print("Original Columns in Fertilizer Dataset:", fertilizer_data.columns)

# # Clean and rename Fertilizer Dataset columns
# fertilizer_data.rename(columns={
#     'Temparature': 'Temperature', 
#     'Humidity ': 'Humidity', 
#     'Moisture': 'Moisture', 
#     'Soil Type': 'Soil Type',
#     'Crop Type': 'Crop Type', 
#     'Nitrogen': 'Nitrogen', 
#     'Potassium': 'Potassium', 
#     'Phosphorous': 'Phosphorous', 
#     'Fertilizer Name': 'Fertilizer Name'
# }, inplace=True)

# print("Cleaned Fertilizer Dataset Columns:")
# print(fertilizer_data.columns)

# # Inspect unique values in Soil Type and Crop Type before mapping
# print("Unique values in Soil Type before mapping:", fertilizer_data['Soil Type'].unique())
# print("Unique values in Crop Type before mapping:", fertilizer_data['Crop Type'].unique())

# # Encode categorical columns
# soil_type_mapping = {'Sandy': 0, 'Clayey': 1, 'Loamy': 2, 'Silt': 3}
# crop_type_mapping = {'Wheat': 0, 'Rice': 1, 'Maize': 2, 'Barley': 3}

# fertilizer_data['Soil Type'] = fertilizer_data['Soil Type'].map(soil_type_mapping)
# fertilizer_data['Crop Type'] = fertilizer_data['Crop Type'].map(crop_type_mapping)

# # Handle missing or unrecognized values
# fertilizer_data['Soil Type'] = fertilizer_data['Soil Type'].fillna(0)  # Default to 'Sandy'
# fertilizer_data['Crop Type'] = fertilizer_data['Crop Type'].fillna(0)  # Default to 'Wheat'

# # Verify the encoding
# print(fertilizer_data[['Soil Type', 'Crop Type']].head())
# print("Unique values in Soil Type after encoding:", fertilizer_data['Soil Type'].unique())
# print("Unique values in Crop Type after encoding:", fertilizer_data['Crop Type'].unique())

# # Define features and target for Fertilizer Recommendation
# fertilizer_features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
# fertilizer_target = 'Fertilizer Name'

# X_fertilizer = fertilizer_data[fertilizer_features]
# y_fertilizer = fertilizer_data[fertilizer_target]

# # Split Fertilizer Data into Training and Testing Sets
# X_train_fertilizer, X_test_fertilizer, y_train_fertilizer, y_test_fertilizer = train_test_split(
#     X_fertilizer, y_fertilizer, test_size=0.2, random_state=42
# )

# # Train Fertilizer Recommendation Models
# dt_model_fertilizer = DecisionTreeClassifier(max_depth=5, random_state=42)
# dt_model_fertilizer.fit(X_train_fertilizer, y_train_fertilizer)

# rf_model_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model_fertilizer.fit(X_train_fertilizer, y_train_fertilizer)

# # Evaluate Fertilizer Recommendation Models
# fertilizer_dt_predictions = dt_model_fertilizer.predict(X_test_fertilizer)
# fertilizer_rf_predictions = rf_model_fertilizer.predict(X_test_fertilizer)

# print("Fertilizer Decision Tree Accuracy:", accuracy_score(y_test_fertilizer, fertilizer_dt_predictions))
# print("Fertilizer Random Forest Accuracy:", accuracy_score(y_test_fertilizer, fertilizer_rf_predictions))

# # Save the trained Random Forest model
# with open('random_forest_model_fertilizer.pkl', 'wb') as file:
#     pickle.dump(rf_model_fertilizer, file)

# print("Model saved as 'random_forest_model_fertilizer.pkl'")
# # Save the trained Random Forest model for crop recommendation
# with open('random_forest_model_crop.pkl', 'wb') as file:
#     pickle.dump(rf_model_crop, file)

# print("Crop recommendation model saved as 'random_forest_model_crop.pkl'")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load Crop Recommendation Data
crop_data = pd.read_csv("E:/crop1/Crop_recommendation.csv")
print("Columns in Crop Dataset:", crop_data.columns)

# Load Fertilizer Recommendation Data
fertilizer_data = pd.read_csv("E:/crop1/Fertilizer Prediction.csv")
print("Original Columns in Fertilizer Dataset:", fertilizer_data.columns)

# Clean and rename Fertilizer Dataset columns
fertilizer_data.rename(columns={
    'Temparature': 'Temperature', 
    'Humidity ': 'Humidity', 
    'Moisture': 'Moisture', 
    'Soil Type': 'Soil Type',
    'Crop Type': 'Crop Type', 
    'Nitrogen': 'Nitrogen', 
    'Potassium': 'Potassium', 
    'Phosphorous': 'Phosphorous', 
    'Fertilizer Name': 'Fertilizer Name'
}, inplace=True)

print("Cleaned Fertilizer Dataset Columns:", fertilizer_data.columns)

# Encode categorical columns in Fertilizer Data
soil_type_mapping = {'Sandy': 0, 'Clayey': 1, 'Loamy': 2, 'Silt': 3}
crop_type_mapping = {'Wheat': 0, 'Rice': 1, 'Maize': 2, 'Barley': 3}

fertilizer_data['Soil Type'] = fertilizer_data['Soil Type'].map(soil_type_mapping)
fertilizer_data['Crop Type'] = fertilizer_data['Crop Type'].map(crop_type_mapping)

# Define features and target for Fertilizer Recommendation
fertilizer_features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
fertilizer_target = 'Fertilizer Name'

X_fertilizer = fertilizer_data[fertilizer_features]
y_fertilizer = fertilizer_data[fertilizer_target]

# Split Fertilizer Data
X_train_fertilizer, X_test_fertilizer, y_train_fertilizer, y_test_fertilizer = train_test_split(
    X_fertilizer, y_fertilizer, test_size=0.2, random_state=42
)

# Train and Evaluate Fertilizer Models
rf_model_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_fertilizer.fit(X_train_fertilizer, y_train_fertilizer)

fertilizer_predictions = rf_model_fertilizer.predict(X_test_fertilizer)
print("Fertilizer Random Forest Accuracy:", accuracy_score(y_test_fertilizer, fertilizer_predictions))

# Save Fertilizer Model
with open('random_forest_model_fertilizer.pkl', 'wb') as file:
    pickle.dump(rf_model_fertilizer, file)
print("Fertilizer recommendation model saved as 'random_forest_model_fertilizer.pkl'")

# Define features and target for Crop Recommendation
crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
crop_target = 'label'

X_crop = crop_data[crop_features]
y_crop = crop_data[crop_target]

# Split Crop Data
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

# Train and Evaluate Crop Model
rf_model_crop = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_crop.fit(X_train_crop, y_train_crop)

crop_predictions = rf_model_crop.predict(X_test_crop)
print("Crop Recommendation Random Forest Accuracy:", accuracy_score(y_test_crop, crop_predictions))

# Save Crop Model
with open('random_forest_model_crop.pkl', 'wb') as file:
    pickle.dump(rf_model_crop, file)
print("Crop recommendation model saved as 'random_forest_model_crop.pkl'")