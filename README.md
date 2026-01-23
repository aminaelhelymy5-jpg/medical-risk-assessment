# Medical Risk Assessment 🩺

AI-powered interactive app that predicts the risk of **Heart Disease**, **Diabetes**, and **Hypertension** using **Logistic Regression** and **Streamlit**.  

This project is designed for educational purposes and showcases **clinical risk prediction**, **data visualization**, and **interactive dashboards**.

---

## Features

- Predict risk of **Heart Disease**, **Diabetes**, and **Hypertension** based on patient input.
- Visualize **feature contributions** to understand which factors increase or decrease risk.
- Interactive **Streamlit interface** for easy input and real-time predictions.
- Data preprocessing with **SimpleImputer**, scaling with **StandardScaler**.
- Trained using **Logistic Regression** for explainable AI.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – Interactive web app
- **scikit-learn** – Machine Learning (Logistic Regression, preprocessing)
- **pandas** – Data handling
- **Plotly Express** – Interactive charts

---

## Data Sources

This project uses publicly available datasets for each medical condition:

1. **Heart Disease** – [UCI Heart Disease Dataset](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/data/heart-disease.csv?utm_source=chatgpt.com)  
   - Contains patient clinical data such as age, blood pressure, cholesterol, and ECG results.

2. **Diabetes** – [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv)  
   - Includes features like glucose, blood pressure, BMI, insulin, age, and diabetes pedigree function.

3. **Hypertension** – [Hypertension Clinical Dataset](https://www.kaggle.com/datasets/sumedh1507/hypertension-dataset?resource=download) 
   - Clinical data including age, BMI, salt intake, exercise, family history, and blood pressure records.

> ⚠️ Note: All datasets are used for **educational purposes only** and may not represent complete clinical scenarios.
