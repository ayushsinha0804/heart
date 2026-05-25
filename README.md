# 🫀 Heart Disease Risk Predictor

A machine learning web app that predicts the likelihood of heart disease based on clinical parameters, built with **Streamlit** and **Random Forest**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.4-orange?style=flat-square)

---

## 🚀 Live Demo
👉 [Click here to try the app](#) *(add your Streamlit Cloud URL here)*

---

## 📋 Features
- Input 13 clinical parameters via an intuitive UI
- Instant risk prediction using a trained Random Forest model
- Displays model confidence and probability breakdown
- Clean, responsive design

## 🧠 Model
- **Algorithm:** Random Forest Classifier (100 estimators, max depth 7)
- **Dataset:** Cleveland Heart Disease dataset (303 records)
- **Accuracy:** ~85% on test data
- **Features:** Age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, ST slope, vessels colored, thalassemia

---

## 🗂️ Project Structure
```
heart-disease-predictor/
├── app.py                  # Streamlit app
├── train_model.py          # Model training script
├── heart_disease_data.csv  # Dataset
├── requirements.txt        # Dependencies
├── models/
│   ├── heart_model.pkl     # Trained model
│   └── scaler.pkl          # Feature scaler
└── README.md
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/heart-disease-predictor.git
cd heart-disease-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (first time only)
python train_model.py

# 4. Launch the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — done!

---

## ⚕️ Disclaimer
This tool is for **educational purposes only** and is not a substitute for professional medical advice. Always consult a qualified healthcare provider.
