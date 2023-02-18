import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import Functions

stop_words = stopwords.words('english') # defining stop_words
stop_words.remove('not') # removing not from the stop_words list as it contains value in negative movies
lemmatizer = WordNetLemmatizer()
LG_L2_TFIDF_clf = pickle.load(open('model1.pkl', 'rb'))
NB_TFIDF_clf = pickle.load(open('model2.pkl', 'rb'))
NB_BOW_clf = pickle.load(open('model3.pkl', 'rb'))
tf_vectorizer = pickle.load(open('tf_vectorizer.pkl', 'rb'))
BOW_vectorizer = pickle.load(open('BOW_vectorizer.pkl', 'rb'))

model = None










st.title('Sentiment Analysis By Abdelrahman Ragab')
User_review = st.text_input('Enter your Movie sentance Review: ', '')

option = st.selectbox('Choose Model to get the output with',
     ('Logistic regression-L2 with tf-idf', 'Naive bayes with tf-idf'))

st.write('You selected:', option)

if(option =='Logistic regression-L2 with tf-idf'):
    model = LG_L2_TFIDF_clf
else:
    model = NB_TFIDF_clf

    
if(User_review ==''):
    prediction_class = " You didn't enter anything yet"
    st.write('This review is --->', :red[prediction_class])
else:
    prediction_class = Functions.predict_sent(User_review,tf_vectorizer,model)
    st.write('This review is ---> ', prediction_class)




