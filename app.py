import streamlit as st
import numpy as np
import joblib

import warnings
warnings.filterwarnings("ignore")

model= joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")

study_hours =st.slider("study hours per day", 0.0, 12.0, 2.0)
attendence = st.slider("attendence percentage", )
mental_health = st.slider("mental health rating", 1,10,5)
sleep_hours = st.slider("sleep hours per night", 0.0, 12.0, 7.0)
part_time_job=st.selectbox("part-time job", ["No", "Yes"] )

ptj_encoded = 1 if part_time_job == "Yes" else 0
 
if st.button("predict exam score"):
    input_data = np.array([[study_hours,attendence,mental_health,sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)[0]

    prediction =max(0, min(100,prediction))

    st.success(f" predicted exam score: {prediction:.2f}")
    


