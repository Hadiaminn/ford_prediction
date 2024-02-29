import streamlit as st
import pandas as pd
import pickle

st.write("""
# Ford Car Price Prediction App

This app predicts the **Price** for type of advertising stratergy!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    year = st.sidebar.slider('Year', 1990.0, 2020.0, 1000.0) #All Float
    engineSize = st.sidebar.slider('Engine Size', 0.0, 5.0, 1.0)
    mileage = st.sidebar.slider('Mileage', 1.0, 177644.0, 15.0)
    data = {'Year': year,
            'Engine Size': engineSize,
            'Mileage': mileage}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("ford_prediction_model.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
