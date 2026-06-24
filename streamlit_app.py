%%writefile streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Teenage Student Dropout Risk Prediction")
st.write(
    "This app predicts a student's dropout risk using a trained XGBoost model, "
    "based on demographic, academic, and behavioral factors. "
    "Fill in the details on the left and click **Predict**."
)

# ----------------------------------------------------------------------
# Load model + expected column order (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("xg_model_pipe.pkl")
    feature_columns = model.named_steps['standardscaler'].feature_names_in_
    return model, feature_columns

model, feature_columns = load_model()

# ----------------------------------------------------------------------
# Sidebar inputs — grouped to match the original dataset's columns
# ----------------------------------------------------------------------
st.sidebar.header("Student Information")

with st.sidebar.expander("Demographics", expanded=True):
    school = st.selectbox("School", ["GP", "MS"])
    gender = st.selectbox("Gender", ["F", "M"])
    age = st.slider("Age", 15, 19, 17)
    address = st.selectbox("Address Type", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")
    family_size = st.selectbox("Family Size", ["GT3", "LE3"], format_func=lambda x: "Greater than 3" if x == "GT3" else "3 or fewer")
    parental_status = st.selectbox("Parental Status", ["T", "A"], format_func=lambda x: "Living together" if x == "T" else "Apart")

with st.sidebar.expander("Family Background"):
    mother_education = st.slider("Mother's Education (0=none, 4=higher)", 0, 4, 2)
    father_education = st.slider("Father's Education (0=none, 4=higher)", 0, 4, 2)
    mother_job = st.selectbox("Mother's Job", ["at_home", "health", "other", "services", "teacher"])
    father_job = st.selectbox("Father's Job", ["at_home", "health", "other", "services", "teacher"])
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])
    family_support = st.selectbox("Family Educational Support", ["yes", "no"])
    family_relationship = st.slider("Quality of Family Relationship (1=very bad, 5=excellent)", 1, 5, 4)

with st.sidebar.expander("School Life"):
    reason = st.selectbox("Reason for Choosing School", ["course", "home", "reputation", "other"])
    school_support = st.selectbox("Extra School Support", ["yes", "no"])
    extra_paid_class = st.selectbox("Extra Paid Classes", ["yes", "no"])
    travel_time = st.slider("Travel Time (1=<15min, 4=>1hr)", 1, 4, 1)
    study_time = st.slider("Weekly Study Time (1=<2hrs, 4=>10hrs)", 1, 4, 2)
    number_of_failures = st.slider("Past Class Failures", 0, 4, 0)
    wants_higher_education = st.selectbox("Wants Higher Education", ["yes", "no"])

with st.sidebar.expander("Lifestyle"):
    extra_curricular = st.selectbox("Extra-Curricular Activities", ["yes", "no"])
    attended_nursery = st.selectbox("Attended Nursery", ["yes", "no"])
    internet_access = st.selectbox("Internet Access at Home", ["yes", "no"])
    in_relationship = st.selectbox("In a Relationship", ["yes", "no"])
    free_time = st.slider("Free Time After School (1=very low, 5=very high)", 1, 5, 3)
    going_out = st.slider("Going Out with Friends (1=very low, 5=very high)", 1, 5, 3)
    weekday_alcohol = st.slider("Weekday Alcohol Consumption (1=very low, 5=very high)", 1, 5, 1)
    weekend_alcohol = st.slider("Weekend Alcohol Consumption (1=very low, 5=very high)", 1, 5, 1)
    health_status = st.slider("Health Status (1=very bad, 5=very good)", 1, 5, 4)

with st.sidebar.expander("Academics"):
    number_of_absences = st.slider("Number of Absences", 0, 100, 4)
    grade_1 = st.slider("Grade 1 (0-20)", 0, 20, 10)
    grade_2 = st.slider("Grade 2 (0-20)", 0, 20, 10)
    final_grade = st.slider("Final Grade (0-20)", 0, 20, 10)

# ----------------------------------------------------------------------
# Build a single-row raw dataframe with the ORIGINAL column names,
# exactly as they existed before one-hot encoding in the notebook.
# ----------------------------------------------------------------------
raw_input = pd.DataFrame([{
    "School": school,
    "Gender": gender,
    "Age": age,
    "Address": address,
    "Family_Size": family_size,
    "Parental_Status": parental_status,
    "Mother_Education": mother_education,
    "Father_Education": father_education,
    "Mother_Job": mother_job,
    "Father_Job": father_job,
    "Reason_for_Choosing_School": reason,
    "Guardian": guardian,
    "Travel_Time": travel_time,
    "Study_Time": study_time,
    "Number_of_Failures": number_of_failures,
    "School_Support": school_support,
    "Family_Support": family_support,
    "Extra_Paid_Class": extra_paid_class,
    "Extra_Curricular_Activities": extra_curricular,
    "Attended_Nursery": attended_nursery,
    "Wants_Higher_Education": wants_higher_education,
    "Internet_Access": internet_access,
    "In_Relationship": in_relationship,
    "Family_Relationship": family_relationship,
    "Free_Time": free_time,
    "Going_Out": going_out,
    "Weekend_Alcohol_Consumption": weekend_alcohol,
    "Weekday_Alcohol_Consumption": weekday_alcohol,
    "Health_Status": health_status,
    "Number_of_Absences": number_of_absences,
    "Grade_1": grade_1,
    "Grade_2": grade_2,
    "Final_Grade": final_grade,
}])

# Same one-hot encoding scheme as the notebook (drop_first=True == drop="first")
OHE_cols = [
    "School", "Gender", "Address", "Family_Size", "Parental_Status",
    "Mother_Job", "Father_Job", "Reason_for_Choosing_School", "Guardian",
    "School_Support", "Family_Support", "Extra_Paid_Class",
    "Extra_Curricular_Activities", "Attended_Nursery",
    "Wants_Higher_Education", "Internet_Access", "In_Relationship",
]
encoded_input = pd.get_dummies(raw_input, columns=OHE_cols, drop_first=True)

# Align to the exact column set/order the model was trained on.
# Any one-hot column not present for this particular input (e.g. a category
# that wasn't picked) gets filled with 0, which is the correct OHE behavior.
encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)

# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------
st.markdown("---")
predict_clicked = st.button("🔮 Predict Dropout Risk", use_container_width=True)

if predict_clicked:
    prediction = model.predict(encoded_input)[0]
    probability = model.predict_proba(encoded_input)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Dropout — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ Low Risk of Dropout — estimated probability: {probability:.1%}")

    st.progress(float(probability))

    with st.expander("See model input (debug view)"):
        st.dataframe(encoded_input)

st.markdown("___")
st.caption("Built with Streamlit • Model: XGBoost (scale_pos_weight-balanced) • Explainability: SHAP & LIME (see notebook)")
