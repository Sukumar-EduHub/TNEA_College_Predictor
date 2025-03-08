import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("tnea_data.csv")

# Ensure the "Category" and "College" columns exist
if "Category" not in df.columns or "College" not in df.columns:
    raise KeyError("❌ Error: Required columns ('Category', 'College') are missing in the dataset!")

# Define all possible categories (to avoid unseen label errors)
all_categories = ["OC", "BC", "MBC", "SC", "ST"]  # Ensure all categories are included

# Fit LabelEncoder with all categories (to prevent unseen labels)
category_encoder = LabelEncoder()
category_encoder.fit(all_categories)  # Train on predefined categories

# Encode category column
df["Category"] = category_encoder.transform(df["Category"])

# Encode college names
college_encoder = LabelEncoder()
df["College"] = college_encoder.fit_transform(df["College"])

# Features and target variable
X = df.drop(columns=["College"])  # Features
y = df["College"]  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and encoders
joblib.dump(model, "tnea_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(category_encoder, "category_encoder.pkl")  # Save category encoder
joblib.dump(college_encoder, "college_encoder.pkl")    # Save college encoder

print("✅ Model trained and saved successfully!")
