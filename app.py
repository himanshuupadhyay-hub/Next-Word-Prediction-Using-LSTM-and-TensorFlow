import streamlit as st
import pickle
import numpy as np
import os 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "nextword_model.h5")

model=load_model(model_path)

tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

with open(tokenizer_path, "rb") as file:
    tokenizer = pickle.load(file)

reverse_index={idx:word for word,idx in tokenizer.word_index.items()}
 
max_len=44

def generate_text(seed_text,num_words=10):
    text=seed_text
    for _ in range(num_words):
        seq=tokenizer.texts_to_sequences([text])[0]
        padded=pad_sequences([seq],maxlen=max_len,padding='pre')
        preds=model.predict(padded,verbose=0)
        pos=np.argmax(preds)
        next_word=reverse_index.get(pos," ")
        text+=" "+next_word
    return text



st.title("Next Word Predictor Using Deep Learning")

seed_text=st.text_input('Enter a Starting Text:','Hello')

num_words=st.slider('Number of words to be generated',1,10,20)

if st.button("Generate"):
    result=generate_text(seed_text,num_words)
    st.write(result)
