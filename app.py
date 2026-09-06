import streamlit as st
import joblib
import pandas as pd

# Load the trained model
# Make sure 'linear_regression_model.sav' is in the same directory as app.py or provide the full path
loaded_model = joblib.load('linear_regression_model.sav')

st.title('Sales Prediction using Linear Regression')
st.write('Enter the advertising budgets for TV, Radio, and Newspaper to predict sales.')

# Create input fields for the features
tv = st.number_input('TV Advertising Budget ($)', min_value=0.0, value=100.0, step=0.1)
radio = st.number_input('Radio Advertising Budget ($)', min_value=0.0, value=20.0, step=0.1)
newspaper = st.number_input('Newspaper Advertising Budget ($)', min_value=0.0, value=30.0, step=0.1)

# Create a DataFrame for the input
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])

# Make prediction when a button is clicked
if st.button('Predict Sales'):
    prediction = loaded_model.predict(input_data)[0]
    st.success(f'Predicted Sales: {prediction:.2f}')
