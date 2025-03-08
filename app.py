import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ✅ Load dataset
DATA_FILE = "tnea_data.csv"

if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset not found! Run `generate_data.py` first.")
    st.stop()

df = pd.read_csv(DATA_FILE)

# ✅ Encode categorical variables
label_encoder_category = LabelEncoder()
df["Category"] = label_encoder_category.fit_transform(df["Category"])

label_encoder_college = LabelEncoder()
df["College"] = label_encoder_college.fit_transform(df["College"])

# ✅ Train-Test Split
X = df[["Physics", "Chemistry", "Mathematics", "Total_Score", "Category"]]
y = df["College"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Model (Only Once)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 🔹 Streamlit UI
st.title("🎓 TNEA College Predictor")

# 📌 User Input
physics = st.number_input("Physics Marks (out of 100)", min_value=0, max_value=100, step=1)
chemistry = st.number_input("Chemistry Marks (out of 100)", min_value=0, max_value=100, step=1)
mathematics = st.number_input("Mathematics Marks (out of 100)", min_value=0, max_value=100, step=1)
total_score = st.number_input("TNEA cutoff Score (out of 300)", min_value=0, max_value=300, step=1)
category = st.selectbox("Category", ["OC", "BC", "MBC", "SC", "ST"])

# 🔍 Predict College
if st.button("Predict College"):
    category_encoded = label_encoder_category.transform([category])[0]
    input_data = np.array([[physics, chemistry, mathematics, total_score, category_encoded]])
    
    predicted_college_code = model.predict(input_data)[0]
    predicted_college = label_encoder_college.inverse_transform([predicted_college_code])[0]

    st.success(f"🎯 Predicted College: **{predicted_college}**")
    st.info(f"📊 Model Accuracy: **{accuracy:.2%}**")
