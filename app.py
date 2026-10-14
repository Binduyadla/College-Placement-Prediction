
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# ==================================================
# COLLEGE PLACEMENT PREDICTION SYSTEM
# Synthetic sample data for demonstration
# ==================================================
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)
NUMERIC = [
    "SSC_Percentage",
    "Intermediate_Percentage",
    "CGPA",
    "Aptitude_Score",
    "Communication_Score",
    "Internships",
    "Projects",
    "Backlogs"
]
FEATURES = NUMERIC + ["Training_Completed"]
@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 1500
    data = pd.DataFrame({
        "SSC_Percentage": rng.uniform(45, 100, n),
        "Intermediate_Percentage": rng.uniform(45, 100, n),
        "CGPA": rng.uniform(4, 10, n),
        "Aptitude_Score": rng.uniform(20, 100, n),
        "Communication_Score": rng.uniform(1, 10, n),
        "Internships": rng.integers(0, 4, n),
        "Projects": rng.integers(0, 6, n),
        "Backlogs": rng.integers(0, 5, n),
        "Training_Completed": rng.choice(
            ["Yes", "No"], n, p=[0.65, 0.35]
        )
    })
    # Artificial labels for a demonstration dataset only
    score = (
        0.07 * (data["SSC_Percentage"] - 60)
        + 0.07 * (data["Intermediate_Percentage"] - 60)
        + 0.9 * (data["CGPA"] - 6)
        + 0.035 * (data["Aptitude_Score"] - 50)
        + 0.20 * (data["Communication_Score"] - 5)
        + 0.35 * data["Internships"]
        + 0.15 * data["Projects"]
        - 0.55 * data["Backlogs"]
        + 0.5 * (data["Training_Completed"] == "Yes").astype(int)
        - 1.0
    )
    p = 1 / (1 + np.exp(-score))
    data["Placement_Status"] = np.where(
        rng.random(n) < p, "Placed", "Not Placed"
    )
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    category_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("category", category_pipe, ["Training_Completed"])
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data["Placement_Status"],
        test_size=0.2,
        random_state=42,
        stratify=data["Placement_Status"]
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return data, model, accuracy
data, model, accuracy = train_model()
# ==================================================
# HEADER
# ==================================================
st.title("🎓 College Placement Prediction System")
st.write(
    "Enter the student's academic and skill details below "
    "to generate a placement prediction."
)
st.info(
    "Demonstration version: the model uses synthetic sample data. "
    "Its predictions and accuracy do not represent real college outcomes."
)
# Student form is the FIRST screen
st.subheader("📝 Student Academic Details")
with st.form("student_details_form"):
    st.markdown("#### Personal Details")
    student_id = st.text_input("Student ID", placeholder="e.g. STU1001")
    st.markdown("#### Academic Performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        ssc = st.number_input(
            "SSC Percentage", 0.0, 100.0, 80.0, step=0.5
        )
    with c2:
        inter = st.number_input(
            "Intermediate / Diploma Percentage",
            0.0, 100.0, 75.0, step=0.5
        )
    with c3:
        cgpa = st.number_input(
            "Current CGPA", 0.0, 10.0, 8.0, step=0.1
        )
    st.markdown("#### Skills and Experience")
    c4, c5, c6 = st.columns(3)
    with c4:
        aptitude = st.number_input(
            "Aptitude Score (0–100)", 0, 100, 75
        )
    with c5:
        communication = st.number_input(
            "Communication Score (1–10)", 1, 10, 7
        )
    with c6:
        training = st.selectbox(
            "Placement Training Completed", ["Yes", "No"]
        )
    c7, c8, c9 = st.columns(3)
    with c7:
        internships = st.number_input(
            "Internships", 0, 20, 1
        )
    with c8:
        projects = st.number_input(
            "Projects Completed", 0, 30, 2
        )
    with c9:
        backlogs = st.number_input(
            "Active Backlogs", 0, 20, 0
        )
    submitted = st.form_submit_button(
        "🔍 PREDICT PLACEMENT",
        use_container_width=True
    )
if submitted:
    if not student_id.strip():
        st.error("Please enter a Student ID.")
    else:
        student = pd.DataFrame([{
            "SSC_Percentage": ssc,
            "Intermediate_Percentage": inter,
            "CGPA": cgpa,
            "Aptitude_Score": aptitude,
            "Communication_Score": communication,
            "Internships": internships,
            "Projects": projects,
            "Backlogs": backlogs,
            "Training_Completed": training
        }])[FEATURES]
        prediction = model.predict(student)[0]
        probabilities = model.predict_proba(student)[0]
        classes = list(model.classes_)
        placement_probability = (
            probabilities[classes.index("Placed")] * 100
        )
        st.divider()
        st.subheader("📊 Student Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Student ID", student_id.strip())
        with col2:
            st.metric("Predicted Placement Status", prediction)
        st.metric(
            "Estimated Placement Probability",
            f"{placement_probability:.1f}%"
        )
        st.progress(float(placement_probability / 100))
        st.markdown("#### Submitted Student Details")
        st.dataframe(student, use_container_width=True)
        st.caption(
            "The probability is a model estimate, not a guarantee. "
            "The model must be trained and validated on real, "
            "authorized historical placement data for practical use."
        )
# ==================================================
# SECONDARY ANALYSIS SECTION
# ==================================================
with st.expander("📈 View Dataset and Model Analysis"):
    st.write("Model test accuracy on synthetic data:")
    st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    st.write("Sample placement outcomes")
    fig, ax = plt.subplots()
    data["Placement_Status"].value_counts().plot(
        kind="bar", ax=ax
    )
    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Students")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)
    st.write("Synthetic sample dataset")
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button(
        "Download Sample Dataset CSV",
        data.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_placement_dataset.csv",
        mime="text/csv"
    )
# ==================================================
# BULK STUDENT PREDICTION
# ==================================================
with st.expander("📂 Upload Multiple Students (CSV / Excel)"):
    st.write(
        "Required columns: " + ", ".join(FEATURES)
    )
    template = data[FEATURES].head(5)
    st.download_button(
        "Download Input Template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="student_input_template.csv",
        mime="text/csv"
    )
    uploaded = st.file_uploader(
        "Upload student data",
        type=["csv", "xlsx"]
    )
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                batch = pd.read_csv(uploaded)
            else:
                batch = pd.read_excel(uploaded)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                batch = batch.copy()

                for col in NUMERIC:
                    batch[col] = pd.to_numeric(
                        batch[col], errors="coerce"
                    )
                batch["Training_Completed"] = (
                    batch["Training_Completed"]
                    .astype("string").str.strip().str.title()
                )
                valid = batch[FEATURES].notna().all(axis=1)
                valid &= batch["Training_Completed"].isin(["Yes", "No"])
                valid &= batch["SSC_Percentage"].between(0, 100)
                valid &= batch["Intermediate_Percentage"].between(0, 100)
                valid &= batch["CGPA"].between(0, 10)
                valid &= batch["Aptitude_Score"].between(0, 100)
                valid &= batch["Communication_Score"].between(1, 10)
                for col in ["Internships", "Projects", "Backlogs"]:
                    valid &= batch[col].ge(0)
                results = batch.loc[valid].copy()
                if results.empty:
                    st.error("No valid rows found. Please check the input.")
                else:
                    results["Predicted_Placement_Status"] = (
                        model.predict(results[FEATURES])
                    )
                    probabilities = model.predict_proba(results[FEATURES])
                    classes = list(model.classes_)
                    results["Placement_Probability_Percent"] = (
                        probabilities[:, classes.index("Placed")] * 100
                    ).round(2)
                    st.success(
                        f"Predictions completed for {len(results)} students. "
                        f"Invalid rows skipped: {int((~valid).sum())}."
                    )
                    st.dataframe(results, use_container_width=True)
                    st.download_button(
                        "Download Prediction Results",
                        results.to_csv(index=False).encode("utf-8"),
                        file_name="placement_prediction_results.csv",
                        mime="text/csv"
                    )
        except Exception as error:
            st.error(f"Unable to read the file: {error}")