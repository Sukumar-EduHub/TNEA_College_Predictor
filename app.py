import pandas as pd
import numpy as np
import streamlit as st
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 📌 File Paths
DATA_FILE = "tnea_data.csv"
MODEL_FILE = "tnea_model.pkl"
ENCODER_CATEGORY_FILE = "category_encoder.pkl"
ENCODER_COLLEGE_FILE = "college_encoder.pkl"

# ✅ Load dataset
if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset not found! Please provide 'tnea_data.csv'.")
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

# ✅ Train Model (If not already saved)
if not os.path.exists(MODEL_FILE):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # ✅ Save Model & Encoders
    joblib.dump(model, MODEL_FILE)
    joblib.dump(label_encoder_category, ENCODER_CATEGORY_FILE)
    joblib.dump(label_encoder_college, ENCODER_COLLEGE_FILE)
else:
    # ✅ Load Saved Model & Encoders
    model = joblib.load(MODEL_FILE)
    label_encoder_category = joblib.load(ENCODER_CATEGORY_FILE)
    label_encoder_college = joblib.load(ENCODER_COLLEGE_FILE)

# ✅ Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 🎨 Streamlit UI Design
st.title("🎓 TNEA College Predictor")

st.markdown(
    """
    ## 📌 Enter Your Marks
    Fill in your Physics, Chemistry, and Mathematics marks to calculate your **TNEA Cutoff Score**.
    """
)

# 📌 User Input
col1, col2, col3 = st.columns(3)

with col1:
    math_marks = st.number_input("📘 Mathematics Marks", min_value=0, max_value=100, step=1)
with col2:
    physics_marks = st.number_input("🔬 Physics Marks", min_value=0, max_value=100, step=1)
with col3:
    chemistry_marks = st.number_input("🧪 Chemistry Marks", min_value=0, max_value=100, step=1)

category = st.selectbox("📌 Category", ["OC", "BC", "MBC", "SC", "ST"])

# ✅ Automatically Calculate Cutoff Score
cutoff_score = (math_marks / 2) + (physics_marks / 4) + (chemistry_marks / 4)

# ✅ Display Cutoff Score in Proper Format
st.markdown(f"""
    <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2E86C1;">
        ✅ Cut Off = {cutoff_score:.2f} / 200
    </div>
    """, unsafe_allow_html=True)

# 🔍 Predict College
if st.button("🎯 Predict College"):
    try:
        category_encoded = label_encoder_category.transform([category])[0]
        input_data = np.array([[physics_marks, chemistry_marks, math_marks, cutoff_score, category_encoded]])
        
        predicted_college_code = model.predict(input_data)[0]
        predicted_college = label_encoder_college.inverse_transform([predicted_college_code])[0]

        st.success(f"🎓 **Predicted College: {predicted_college}**")
        st.info(f"📊 **Model Accuracy: {accuracy:.2%}**")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
