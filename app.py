import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Career Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load model
model = joblib.load("career_model.pkl")

st.title("🎓 Career Recommendation System")
st.write("Enter your details to get the best career recommendation.")

# Input fields
programming = st.slider("Programming Skill", 1, 10, 5)
math = st.slider("Math Skill", 1, 10, 5)
communication = st.slider("Communication Skill", 1, 10, 5)
logical = st.slider("Logical Thinking", 1, 10, 5)
cgpa = st.slider("CGPA", 5.0, 10.0, 7.0)
projects = st.slider("Projects Completed", 0, 10, 2)

st.subheader("Interests")
interest_coding = st.checkbox("Coding")
interest_data = st.checkbox("Data Science")
interest_appdev = st.checkbox("App Development")
interest_security = st.checkbox("Cybersecurity")

favorite_subject = st.selectbox(
    "Favorite Subject",
    ["AI", "DBMS", "Networks", "Cybersecurity", "Web", "Maths", "DSA"]
)

# Convert checkboxes to 0/1
interest_coding = int(interest_coding)
interest_data = int(interest_data)
interest_appdev = int(interest_appdev)
interest_security = int(interest_security)

# Create input dataframe
user_input = pd.DataFrame([{
    "programming_skill": programming,
    "math_skill": math,
    "communication_skill": communication,
    "logical_thinking": logical,
    "cgpa": cgpa,
    "projects_count": projects,
    "interest_coding": interest_coding,
    "interest_data": interest_data,
    "interest_appdev": interest_appdev,
    "interest_security": interest_security,
    "favorite_subject": favorite_subject
}])

if st.button("🔮 Predict Career"):
    prediction = model.predict(user_input)[0]
    st.success(f"🎯 **Recommended Career: {prediction}**")
