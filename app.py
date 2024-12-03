# -*- coding: utf-8 -*-
"""
#######
#Group 2
#Von Ezekiel Dela Cruz, Manaois- 301287836
# Damini, Bhoi- 301369239
# Guiqin, Li- 301298267
#Akeem, Pingue -301178051
#Abduljelili, Umaru - 301318075
########
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading the saved models

heart_disease_model = pickle.load(open('D:/SEMESTER-4/377-AI/Assignment/heart_disease_model.pkl', 'rb'))
hepatitis_disease_model = pickle.load(open('D:/SEMESTER-4/377-AI/Assignment/hepatitis_disease_model.pkl', 'rb'))
diabetes_disease_model = pickle.load(open('D:/SEMESTER-4/377-AI/Assignment/diabetes_model.pkl', 'rb'))


# sidebar for navigate

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Heart Disease Prediction',
                           'Hepatitis Disease Prediction',
                           'Diabetes Disease Prediction'],
                            default_index = 0 )
    
# Heart Prediction page
if (selected == 'Heart Disease Prediction'):
    st.title ('Heart Disease Prediction')
    
    # Create input fields for each feature required by the heart disease model
    age = st.number_input('Age', min_value=0, max_value=120, value=30, step=1)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.selectbox('Chest Pain Type', options=[1, 2, 3, 4])
    trestbps = st.number_input('Resting Blood Pressure', min_value=90, max_value=200, value=120, step=1)
    chol = st.number_input('Serum Cholestrol in mg/dl', min_value=100, max_value=600, value=200, step=1)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=100, step=1)
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of the peak exercise ST segment', options=[1, 2, 3])
    ca = st.number_input('Number of major vessels colored by flourosopy', min_value=0, max_value=3, step=1)
    thal = st.selectbox('Thal', options=[3, 6, 7], format_func=lambda x: 'Normal' if x == 3 else 'Fixed defect' if x == 6 else 'Reversible defect')

    # Button to predict
    if st.button('Predict Heart Disease'):
        # Make prediction
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_disease_model.predict(features)
        # Display prediction
        result = 'Positive for Heart Disease' if prediction[0] == 1 else 'Negative for Heart Disease'
        st.write('Prediction: ', result)
    
# Hepatitis Disease Prediction page    
if (selected == 'Hepatitis Disease Prediction'):
    st.title ('Hepatitis Disease Prediction')
    
    # Input fields based on dataset columns
    age = st.number_input('Age', min_value=0, max_value=100, value=40, step=1)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    steroid = st.selectbox('Steroid', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    antivirals = st.selectbox('Antivirals', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    fatigue = st.selectbox('Fatigue', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    malaise = st.selectbox('Malaise', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    anorexia = st.selectbox('Anorexia', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    liver_big = st.selectbox('Liver Big', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    liver_firm = st.selectbox('Liver Firm', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    spleen_palpable = st.selectbox('Spleen Palpable', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    spiders = st.selectbox('Spiders', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    ascites = st.selectbox('Ascites', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    varices = st.selectbox('Varices', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    bilirubin = st.number_input('Bilirubin', min_value=0.0, max_value=20.0, value=1.0, step=0.1)
    alk_phosphate = st.number_input('Alkaline Phosphate', min_value=20, max_value=300, value=85, step=1)
    sgot = st.number_input('SGOT', min_value=10, max_value=500, value=45, step=1)
    albumin = st.number_input('Albumin', min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    protime = st.number_input('Protime', min_value=10, max_value=100, value=35, step=1)
    histology = st.selectbox('Histology', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

   # Button to predict
    if st.button('Predict Hepatitis'):
       # Prepare features for prediction
       features = np.array([[age, sex, steroid, antivirals, fatigue, malaise, anorexia, liver_big, liver_firm,
                             spleen_palpable, spiders, ascites, varices, bilirubin, alk_phosphate, sgot, albumin, protime, histology]])
       prediction = hepatitis_disease_model.predict(features)
       # Display prediction
       result = 'Positive for Hepatitis' if prediction[0] == 1 else 'Negative for Hepatitis'
       st.write('Prediction: ', result)
       
# Diabetes Prediction page
if (selected == 'Diabetes Disease Prediction'):
    st.title ('Diabetes Disease Prediction')
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=6, step=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=148, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=140, value=72, step=1)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=35, step=1)
    insulin = st.number_input('Insulin', min_value=0, max_value=846, value=0, step=1)  # Adjust default if needed
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=33.6, step=0.1)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.627, step=0.001)
    age = st.number_input('Age', min_value=0, max_value=100, value=50, step=1)

   # Button to predict
    if st.button('Predict Diabetes'):
       # Prepare features for prediction
       features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
       prediction = diabetes_disease_model.predict(features)
       # Display prediction
       result = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'
       st.write('Prediction: ', result)