import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Medical Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

# =========================
# SIDEBAR – DISEASE SELECTION
# =========================
st.sidebar.title("🧬 Disease Selection")

disease = st.sidebar.selectbox(
    "Select Disease",
    ["Heart Disease", "Diabetes", "Hypertension"]
)

# ======================================================
# ================= HEART DISEASE ======================
# ======================================================
if disease == "Heart Disease":

    # =========================
    # LOAD DATA
    # =========================
    @st.cache_data
    def load_data():
        df = pd.read_csv("heart-disease.csv")
        df["target"] = 1 - df["target"]
        return df

    data = load_data()

    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # TRAIN MODEL
    # =========================
    @st.cache_resource
    def train_model():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    model = train_model()

    # =========================
    # HEADER
    # =========================
    st.markdown("""
    # 🩺 Cardiovascular Risk Assessment
    ### AI-assisted clinical decision support

    This tool estimates **cardiovascular disease risk**  
    using a **Logistic Regression model** trained on clinical data.
    """)
    st.divider()

    # =========================
    # SIDEBAR – PATIENT DATA
    # =========================
    st.sidebar.header("🧑‍⚕️ Patient Information")

    age = st.sidebar.slider("Age", 20, 80, 50)

    sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex == "Female" else 1

    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)

    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200)

    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 0 if fbs == "No" else 1

    restecg = st.sidebar.selectbox(
        "Rest ECG",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)

    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)

    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 0 if exang == "No" else 1

    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

    slope = st.sidebar.selectbox(
        "Slope of ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)

    ca = st.sidebar.selectbox("Number of Major Vessels", ["0", "1", "2", "3"])
    ca = int(ca)

    thal = st.sidebar.selectbox(
        "Thalassemia",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )
    thal = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

    # =========================
    # PREDICTION
    # =========================
    if st.button("🔍 Predict Cardiovascular Risk"):

        patient_data = pd.DataFrame([[ 
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]], columns=X.columns)

        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0][1]

        # =========================
        # RISK LEVEL
        # =========================
        if probability >= 0.8:
            risk = "HIGH 🔴"
            color = "#E74C3C"
        elif probability >= 0.5:
            risk = "MEDIUM 🟠"
            color = "#F1C40F"
        else:
            risk = "LOW 🟢"
            color = "#2ECC71"

        st.subheader("🧾 Clinical Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnosis", "Heart Disease" if prediction == 1 else "No Heart Disease")
        c2.metric("Risk Probability", f"{probability:.2%}")
        c3.markdown(f"<h3 style='color:{color}'>{risk}</h3>", unsafe_allow_html=True)

        st.divider()

        # =========================
        # EXPLANATION
        # =========================
        st.subheader("🧠 Clinical Interpretation")

        coef = pd.Series(
            model.named_steps["model"].coef_[0],
            index=X.columns
        )

        scaled = model.named_steps["scaler"].transform(patient_data)[0]
        contributions = (coef * scaled).sort_values(ascending=False)

        st.markdown("### Factors increasing risk")
        for f, v in contributions.head(5).items():
            if v > 0:
                st.write(f"- **{f}** contributes positively to cardiovascular risk")

        st.markdown("### Protective / lower-risk factors")
        for f, v in contributions.tail(5).items():
            if v < 0:
                st.write(f"- **{f}** is within a protective clinical range")

        # =========================
        # GRAPH
        # =========================
        st.subheader("📊 Feature Contribution Analysis")

        graph_data = contributions.reset_index()
        graph_data.columns = ["Feature", "Impact"]

        fig = px.bar(
            graph_data,
            x="Impact",
            y="Feature",
            orientation="h",
            color="Impact",
            color_continuous_scale=["#2ECC71", "#F1C40F", "#E74C3C"]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ Educational use only – not a medical diagnosis.")
# ======================================================
# ==================== DIABETES ========================
# ======================================================
elif disease == "Diabetes":

    st.title("🩸 Diabetes Risk Assessment")
    st.divider()

    # =========================
    # LOAD DATA
    # =========================
    @st.cache_data
    def load_diabetes_data():
        df = pd.read_csv("diabetes.csv", header=None)
        df.columns = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome"
        ]
        return df

    data = load_diabetes_data()

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # TRAIN MODEL
    # =========================
    @st.cache_resource
    def train_diabetes_model():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    model = train_diabetes_model()

    # =========================
    # SIDEBAR – PATIENT DATA
    # =========================
    st.sidebar.header("🧑‍⚕️ Patient Information (Diabetes)")

    preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose (mg/dL)", 50, 250, 120)
    bp = st.sidebar.slider("Blood Pressure (mm Hg)", 40, 140, 70)
    skin = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin (μU/mL)", 0, 900, 80)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.slider("Age", 10, 90, 40)

    # =========================
    # PREDICTION
    # =========================
    if st.button("🔍 Predict Diabetes Risk"):

        patient_data = pd.DataFrame([[ 
            preg, glucose, bp, skin,
            insulin, bmi, dpf, age
        ]], columns=X.columns)

        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0][1]

        # =========================
        # RISK LEVEL
        # =========================
        if probability >= 0.75:
            risk = "HIGH RISK 🔴"
            color = "#E74C3C"
        elif probability >= 0.45:
            risk = "MODERATE RISK 🟠"
            color = "#F1C40F"
        else:
            risk = "LOW RISK 🟢"
            color = "#2ECC71"

        # =========================
        # RESULT
        # =========================
        st.subheader("🧾 Clinical Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnosis", "Diabetes Detected" if prediction == 1 else "No Diabetes")
        c2.metric("Risk Probability", f"{probability:.2%}")
        c3.markdown(f"<h3 style='color:{color}'>{risk}</h3>", unsafe_allow_html=True)

        st.divider()

        # =========================
        # EXPLANATION (WHY LOW / HIGH)
        # =========================
        st.subheader("🧠 Clinical Interpretation")

        coef = pd.Series(
            model.named_steps["model"].coef_[0],
            index=X.columns
        )

        scaled = model.named_steps["scaler"].transform(patient_data)[0]
        contributions = (coef * scaled).sort_values(ascending=False)

        st.markdown("### 🔴 Factors increasing diabetes risk")
        for f, v in contributions.head(5).items():
            if v > 0:
                st.write(f"- **{f}** significantly increases metabolic risk")

        st.markdown("### 🟢 Protective / lower-risk factors")
        for f, v in contributions.tail(5).items():
            if v < 0:
                st.write(f"- **{f}** is within a protective clinical range")

        # =========================
        # GRAPH
        # =========================
        st.subheader("📊 Feature Contribution Analysis")

        graph_data = contributions.reset_index()
        graph_data.columns = ["Clinical Feature", "Impact on Risk"]

        fig = px.bar(
            graph_data,
            x="Impact on Risk",
            y="Clinical Feature",
            orientation="h",
            color="Impact on Risk",
            color_continuous_scale=["#2ECC71", "#F1C40F", "#E74C3C"],
            title="Clinical Impact of Patient Parameters on Diabetes Risk"
        )

        fig.update_layout(
            xaxis_title="Standardized Impact",
            yaxis_title="Clinical Parameters",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ Educational use only – not a medical diagnosis.")


# =========================
# HYPERTENSION
# =========================
elif disease == "Hypertension":

    st.title("💓 Hypertension Risk Assessment")
    st.write("AI-assisted clinical hypertension risk prediction")
    st.divider()

    # Load & Clean Data
    @st.cache_data
    def load_hypertension_data():
        df = pd.read_csv("hypertension_dataset.csv")

        df["BP_History"] = df["BP_History"].map({"No": 0, "Yes": 1})
        df["Medication"] = df["Medication"].map({"No": 0, "Yes": 1})
        df["Family_History"] = df["Family_History"].map({"No": 0, "Yes": 1})
        df["Smoking_Status"] = df["Smoking_Status"].map({"No": 0, "Yes": 1})
        df["Exercise_Level"] = df["Exercise_Level"].map({"Low": 0, "Moderate": 1, "High": 2})
        df["Has_Hypertension"] = df["Has_Hypertension"].map({"No": 0, "Yes": 1, "Hypertension": 1})
        return df

    data = load_hypertension_data()
    X = data.drop("Has_Hypertension", axis=1)
    y = data["Has_Hypertension"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Model
    @st.cache_resource
    def train_hypertension_model(X_train, y_train):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    model = train_hypertension_model(X_train, y_train)

    # Sidebar Inputs
    st.sidebar.header("🧑‍⚕️ Patient Information (Hypertension)")
    age = st.sidebar.slider("Age", 18, 90, 45)
    salt = st.sidebar.slider("Salt Intake (g/day)", 1.0, 15.0, 6.0)
    stress = st.sidebar.slider("Stress Level", 0, 10, 4)
    bp_history = st.sidebar.selectbox("Previous BP History", ["No", "Yes"])
    sleep = st.sidebar.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0)
    bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
    medication = st.sidebar.selectbox("On Medication", ["No", "Yes"])
    family = st.sidebar.selectbox("Family History", ["No", "Yes"])
    exercise = st.sidebar.selectbox("Exercise Level", ["Low", "Moderate", "High"])
    smoking = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])

    # Prediction
    if st.button("🔍 Predict Hypertension Risk"):

        patient_data = pd.DataFrame([{
            "Age": age,
            "Salt_Intake": salt,
            "Stress_Score": stress,
            "BP_History": 1 if bp_history == "Yes" else 0,
            "Sleep_Duration": sleep,
            "BMI": bmi,
            "Medication": 1 if medication == "Yes" else 0,
            "Family_History": 1 if family == "Yes" else 0,
            "Exercise_Level": {"Low": 0, "Moderate": 1, "High": 2}[exercise],
            "Smoking_Status": 1 if smoking == "Yes" else 0
        }])

        patient_data = patient_data[X.columns]
        probability = model.predict_proba(patient_data)[0][1]
        prediction = 1 if probability >= 0.5 else 0

        # Risk Level
        if probability >= 0.70:
            risk = "HIGH 🔴"
            color = "#E74C3C"
        elif probability >= 0.40:
            risk = "MEDIUM 🟠"
            color = "#F1C40F"
        else:
            risk = "LOW 🟢"
            color = "#2ECC71"

        # Result
        st.subheader("🧾 Clinical Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnosis", "Hypertension Detected" if prediction == 1 else "No Hypertension")
        c2.metric("Risk Probability", f"{probability:.2%}")
        c3.markdown(f"<h3 style='color:{color}'>{risk}</h3>", unsafe_allow_html=True)
        st.divider()

        # Explanation
        st.subheader("🧠 Why this result?")
        lr_model = model.named_steps["model"]
        scaler = model.named_steps["scaler"]
        imputer = model.named_steps["imputer"]

        patient_imputed = pd.DataFrame(imputer.transform(patient_data), columns=X.columns)
        patient_scaled = scaler.transform(patient_imputed)[0]
        coef = pd.Series(lr_model.coef_[0], index=X.columns)
        contributions = pd.Series(coef.values * patient_scaled, index=X.columns).sort_values(ascending=False)

        # Risk / Protective Features
        RISK_FEATURES = ["Age", "Salt_Intake", "Stress_Score", "BMI", "BP_History", "Family_History", "Smoking_Status"]
        PROTECTIVE_FEATURES = ["Exercise_Level", "Sleep_Duration", "Medication"]

        st.markdown("### 🔴 Factors increasing hypertension risk")
        for f in RISK_FEATURES:
            if contributions[f] > 0:
                st.write(f"- **{f}** increases blood pressure risk")

        st.markdown("### 🟢 Protective / lower-risk factors")
        for f in PROTECTIVE_FEATURES:
            if contributions[f] < 0:
                st.write(f"- **{f}** helps reduce hypertension risk")

        # Graph – Feature Contribution
        st.subheader("📊 Feature Contribution Analysis")
        graph_data = contributions.reset_index()
        graph_data.columns = ["Clinical Feature", "Impact on Risk"]

        fig = px.bar(
            graph_data,
            x="Impact on Risk",
            y="Clinical Feature",
            orientation="h",
            color="Impact on Risk",
            color_continuous_scale=["#2ECC71", "#F1C40F", "#E74C3C"]
        )
        fig.update_layout(xaxis_title="Standardized Impact", yaxis_title="Clinical Parameters", plot_bgcolor="rgba(0,0,0,0)", height=450)
        st.plotly_chart(fig, use_container_width=True, key="hypertension_contrib")

        st.caption("⚠️ Educational use only – not a medical diagnosis.")
