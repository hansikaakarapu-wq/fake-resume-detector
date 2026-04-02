import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

# Page setup
st.set_page_config(page_title="Fake Resume Detection", page_icon="🤖")

# Title
st.title("Fake Resume Profile Detection using ANN")

st.markdown("Enter resume details to check whether it is Real or Fake.")

# Sidebar inputs
st.sidebar.header("Resume Details")

experience = st.sidebar.number_input("Experience (years)", 0)
skills = st.sidebar.number_input("Number of Skills", 0)
education = st.sidebar.selectbox("Education Level", [1,2,3,4])
projects = st.sidebar.number_input("Projects", 0)
certifications = st.sidebar.number_input("Certifications", 0)
gap = st.sidebar.number_input("Employment Gap (years)", 0)

# Prediction
if st.button("Predict"):
    input_data = [[experience, skills, education, projects, certifications, gap]]
    result = model.predict(input_data)

    if result[0] == 0:
        st.success("✅ Real Resume")
    else:
        st.error("⚠️ Fake Resume")