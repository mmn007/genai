import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


model = load_model('imdb_rnn.h5')
word_index = imdb.get_word_index()
reverse_word_index = {v:k for k,v in imdb.get_word_index().items()}

def preprocess(text):
    words = text.lower().split()
    encoded_review = [word_index.get(w, 2) + 3 for w in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def decode(arr):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in arr])

def predict(review):
    pr = preprocess(review)
    #print(pr)
    res = model.predict([pr])
    return 'Positive' if res > 0.5 else 'Negative', res[0][0]


st.title("IMDB Review and Sentiment")
st.write('Enter a movie review to find sentiment!')
ip = st.text_area('Enter a review')
if st.button('Classify'):
    print(ip)
    s, r = predict(ip)
    print(s, r)
    st.write(f'Sentiment is {s}')
