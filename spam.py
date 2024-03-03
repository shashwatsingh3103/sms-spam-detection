import streamlit as str
import pickle
import pandas as pd 
import numpy as np
import requests 
import nltk
nltk.download("punkt")
nltk.download("stopwords")
#preprocessing 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
def trnform(text):
  text= text.lower()
  text= nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  s=[]
  #removing stopwords
  for k in y :
    if k not in stopwords.words("english")  and  string.punctuation:
        # stemming
      ps= PorterStemmer()
      k=ps.stem(k)
      s.append(k)
  return " ".join(s) 


vect=pickle.load(open("vect2.pkl","rb"))
model=pickle.load(open("modelemail.pkl","rb"))

str.title(" Email/Sms spam classfier")
input= str.text_area("enter message")

if str.button("predict"):
 traninput= trnform(input)
 vecinput= vect.transform([traninput])
 pre= model.predict(vecinput)
 if pre==0:
      str.header("Not Spam")
 elif pre==1:
      str.header("Spam")
