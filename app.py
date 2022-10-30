# Core Pkgs
from pandas.core.arrays import categorical
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import os
import joblib

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def display():
    st.title('Stroke Prediction Application')


def load_PreTrainedModelDetails():
    PreTrainedModelDetails = joblib.load(
        'model/stroke_pred_classification.joblib')
    return PreTrainedModelDetails


def prediction(input_df):
    PreTrainedModelDetails = load_PreTrainedModelDetails()

    # Decision Tree Classifier
    DecisionTreeClassifier = PreTrainedModelDetails.get('model')

    # PreFitted Encoder
    PreFittedEncoder = PreTrainedModelDetails.get('encoder')

    # PreFitted Scaler
    PreFittedScaler = PreTrainedModelDetails.get('scaler')

    num_cols = PreTrainedModelDetails.get('num_cols')

    cat_cols = PreTrainedModelDetails.get('cat_cols')

    encoded_cols = PreTrainedModelDetails.get('encoded_cols')

    # train_score = PreTrainedModelDetails.get('train_score')

    # val_score = PreTrainedModelDetails.get('val_score')

    model_acc = PreTrainedModelDetails.get('modelAcc')

    input_df[encoded_cols] = PreFittedEncoder.transform(input_df[cat_cols])
    input_df[num_cols] = PreFittedScaler.transform(input_df[num_cols])

    inputs_for_prediction = input_df[num_cols + encoded_cols]

    prediction = DecisionTreeClassifier.predict(inputs_for_prediction)

    accuracy = model_acc

    if prediction == 0:
        st.success("Patient is not at risk of getting a stroke.")
    else:
        st.warning("Patient is at risk of getting a stroke.")

    st.write("Accuracy of the prediction : {}".format(accuracy))


def get_user_input():
    form = st.form(key='user input form')
    gender = form.radio("Gender", ['Male', 'Female', 'Other'], key='gender')
    age = form.number_input("Age", 1, 120, key='age')
    hypertension = form.radio(
        "Hypertension", ['Yes', 'No'], key='hypertension')
    heart_disease = form.radio(
        "Heart Disease", ['Yes', 'No'], key='heart_disease')
    marital_status = form.radio(
        "Married", ['Married', 'Unmarried'], key='marital_status')
    work_type = form.radio("Work Type", ['Private Sector', 'Government Sector', 'Never Worked',
                                         'Self-employed', 'Children'], key='work_type')
    residence_type = form.radio(
        "Residence Type", ['Urban', 'Rural'], key='residence_type')
    avg_glucose_level = form.number_input(
        "Average Glucose Level", 40.0, 400.0, key='avg_glucose_level')
    bmi = form.number_input("BMI", 10.00, 120.00, key='bmi')
    smoking_status = form.radio("Smoking Status", [
        'Never Smoked', 'Formerly Smoked', 'Smokes', 'Unknown'], key='smoking_status')

    submitButton = form.form_submit_button(label='Predict Stroke Condition')

    if submitButton:
        SingleUserInput = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'marital_status': marital_status,
            'work_type': work_type,
            'residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status

        }

        return pd.DataFrame([SingleUserInput])


def main():
    display()
    input_details_df = get_user_input()

    if input_details_df is not None:
        prediction(input_details_df)


if __name__ == '__main__':
    main()
