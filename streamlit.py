import streamlit as st
import pandas as pd
import numpy as np

st.title("Hello Streamlit!")

st.write("This is a simple text")

df = pd.DataFrame({
    "first_column": [10,20,30,40],
    "second_column": [100,200,300,400]
})

st.write(df)

cdata = pd.DataFrame(np.random.randn(20,3),columns=['a','b','c'])
st.line_chart(cdata)
