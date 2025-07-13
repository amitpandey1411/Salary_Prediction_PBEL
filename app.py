# app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
@st.cache_data
def load_and_train():
    df = pd.read_csv('Salary_Data.csv')
    df.dropna(inplace=True)
    df['Education Level'].replace(["Bachelor's Degree","Master's Degree","phD"],["Bachelor's","Master's","PhD"],inplace=True)
    education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df['Education Level'] = df['Education Level'].map(education_mapping)
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    job_title_count = df['Job Title'].value_counts()
    job_title_edited = job_title_count[job_title_count<=25]
    df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited else x )
    dummies = pd.get_dummies(df['Job Title'], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df.drop('Job Title', axis=1, inplace=True)
    features = df.drop('Salary', axis=1)
    target = df['Salary']
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=20)
    model.fit(x_train, y_train)
    return model, le, education_mapping, dummies

model, le, education_mapping, dummies = load_and_train()

# Streamlit UI
st.title('Salary Prediction App')

age = st.number_input('Age', min_value=18, max_value=70, value=30)
gender = st.selectbox('Gender', le.classes_)
education = st.selectbox('Education Level', list(education_mapping.keys()))
years_exp = st.number_input('Years of Experience', min_value=0, max_value=50, value=5)
job_title = st.selectbox('Job Title', ['Others'] + [col for col in dummies.columns])

def prepare_input():
    input_dict = {
        'Age': age,
        'Gender': le.transform([gender])[0],
        'Education Level': education_mapping[education],
        'Years of Experience': years_exp
    }
    for col in dummies.columns:
        input_dict[col] = 1 if col == job_title else 0
    return pd.DataFrame([input_dict])

if st.button('Predict Salary'):
    input_df = prepare_input()
    prediction = model.predict(input_df)[0]
    st.success(f'Predicted Salary: ${prediction:,.2f}')

st.write('---')
st.write('This app uses a Random Forest model trained on your dataset.')
