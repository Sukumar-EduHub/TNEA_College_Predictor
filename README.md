🎓 TNEA College Predictor

🚀 AI-powered College Prediction for Tamil Nadu Engineering Admissions (TNEA)

📌 About

TNEA College Predictor is a machine learning-based web application built using Python, Streamlit, and Scikit-Learn to help students predict the most suitable engineering college based on their Physics, Chemistry, Mathematics (PCM) scores, cutoff marks, and category.

🛠️ Features

✅ Predicts the best-fit engineering college based on TNEA cutoff marks.

✅ Uses RandomForestClassifier for accurate predictions.

✅ Encodes categorical data (Category & College) to prevent unseen label errors.

✅ Interactive web-based UI using Streamlit.

✅ Persists trained model & encoders using Joblib for faster inference.

✅ Displays model accuracy and prevents missing data issues.

🔧 Technologies Used

Python 🐍

Pandas & NumPy for data handling

Scikit-Learn for machine learning

Streamlit for UI

Joblib for model persistence

🚀 How to Run

1️⃣ Clone the Repository

git clone https://github.com/your-username/TNEA_College_Predictor.git
cd TNEA_College_Predictor

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py

📊 Dataset

The dataset (tnea_data.csv) should include the following columns:

Physics (Marks out of 100)

Chemistry (Marks out of 100)

Mathematics (Marks out of 100)

Total_Score (Cutoff Score out of 300)

Category (OC, BC, MBC, SC, ST)

College (Engineering College Name)

💡 Future Enhancements

🔹 Add more college datasets for better predictions

🔹 Implement additional ML models (XGBoost, SVM) for accuracy comparison

🔹 Improve UI with ranking & filtering features

💪 Contributions & Feedback

🙌 Contributions are welcome! Feel free to fork, improve, and submit a pull request.

🔥 Star ⭐ the repository if you find it useful!

