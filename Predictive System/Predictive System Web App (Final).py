import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('D:/Python Projects/Diabetes Prediction System/Google Colab/trained_model.sav', 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return ("The person is not Diabetic")
    else:
        return ("The person is Diabetic")

# Main UI function
def main():
    st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
    st.title("Diabetes Prediction System")
    st.markdown("---")
    st.markdown("Please enter the following health information to predict Diabetes")

    # Input fields in two columns
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0)
        Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")

    with col2:
        Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0)
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0)
        BMI = st.number_input("BMI (kg/m²)", min_value=0.0, format="%.1f")
        Age = st.number_input("Age (years)", min_value=1, max_value=120)

    # Prediction
    st.markdown("---")
    diagnosis = ""
    if st.button("Get Diabetes Prediction"):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = diabetes_prediction(input_data)
        st.success(diagnosis)

    st.markdown("---")

if __name__ == '__main__':
    main()