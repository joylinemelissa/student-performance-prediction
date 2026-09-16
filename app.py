import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Input and output
X = data[["study_hours", "attendance", "previous_marks", "assignment_score"]]
y = data["final_score"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Title
st.title("Student Performance Prediction System")

st.write("Enter the student's details to predict the final score.")

# User inputs
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous_marks = st.number_input("Previous Marks", min_value=0.0, max_value=100.0)
assignment_score = st.number_input("Assignment Score", min_value=0.0, max_value=100.0)

# Prediction button
if st.button("Predict Final Score"):
    prediction = model.predict([[
        study_hours,
        attendance,
        previous_marks,
        assignment_score
    ]])

    st.success(f"Predicted Final Score: {prediction[0]:.2f}")