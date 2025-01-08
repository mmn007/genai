import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from keras.models import load_model

import streamlit as st

ann = load_model('datasets/ann_model.h5')

with open('datasets/label_encoded_gender.pkl', 'rb') as fp:
    leg = pickle.load(fp)
with open('datasets/ohe_encoded_geo.pkl', 'rb') as fp:
    ohe = pickle.load(fp)
with open('datasets/scaler.pkl', 'rb') as fp:
    scaler = pickle.load(fp)

st.title('Customer Churn prediction')
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', leg.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

ge = ohe.transform([[input_data['Geography']]]).toarray()
ge_df = pd.DataFrame(ge, columns=ohe.get_feature_names_out(['Geography']))
df = pd.DataFrame([input_data])
df.drop(['Geography'], axis=1, inplace=True)

df = pd.concat([df.reset_index(drop=True), ge_df], axis=1)

df['Gender'] = leg.fit_transform(df['Gender'])

scaled = scaler.transform(df)

predictions = ann.predict(scaled)
prediction_proba = predictions[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
