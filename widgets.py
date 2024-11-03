import streamlit as st
import pandas as pd

st.title("Streamlit Text Input")

name = st.text_input("Enter your name")

age = st.slider('Select your age', 0, 100,25)

lg = st.selectbox('Favourite language?', ['Java', 'Python', 'C++', 'Go'])

if name:
    st.write(f'Hello {name} aged {age}')
