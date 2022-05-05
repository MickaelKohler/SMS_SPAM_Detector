import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('popular')
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
import string
import re

# CONFIG

st.set_page_config(page_title="SPAM Explorer",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="auto",
                   )

st.markdown("""
    <style>
    .titre {
        font-size:16px;
        font-weight:normal;
        margin:0px;
    }
    .text {
        font-size:16px;
        font-weight:normal;
        color:lightgray;
    }
    .sous_indice {
        font-size:60px;
        font-weight:bold;
    }
    .indice_total {
        font-size:100px;
        font-weight:bold;
    }
    </style>
    """, unsafe_allow_html=True)


# FONCTIONS

def netto(text):

  text = text.lower()
  text = re.sub(r'[\d]', ' ', text) 
  text = re.sub(r'[^\w\s]', ' ', text) 
  text = "".join([word+' ' for word in text.split() if word not in stopWords])
  text = "".join([lem_adv(word)+' ' for word in text.split()])
  return text

def lem_adv(word):
    from nltk.corpus import wordnet

    lem = WordNetLemmatizer()
    

    # Get the single character pos constant from pos_tag like this:
    pos_label = (pos_tag(word_tokenize(word))[0][1][0]).lower()

    # pos_refs = {'n': ['NN', 'NNS', 'NNP', 'NNPS'],
    #            'v': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    #            'r': ['RB', 'RBR', 'RBS'],
    #            'a': ['JJ', 'JJR', 'JJS']}

    if pos_label in ['a', 's', 'v']: # For adjectives and verbs
         return(lem.lemmatize(word, pos=pos_label))
    else:   # For nouns and everything else as it is the default kwarg
        return(lem.lemmatize(word))

# DATA

data = pd.read_csv("/home/franck/Documents/Notebooks/SPAM_project/data_vect.csv")
data.fillna("empty", inplace=True)


# SIDE BAR

thematiques = [
    'Sci-kit Learn',
    ]

st.sidebar.title('SPAM Explorer')
st.sidebar.title(" ")

section = st.sidebar.radio(
    'Selection : ',
    thematiques)
st.sidebar.subheader(' ')


# MAIN PAGE
vec = CountVectorizer()


X= data.message_lem
y = data['label']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)
vec = CountVectorizer().fit(X_train)
X_train = vec.transform(X_train)
X_test = vec.transform(X_test)  


if section == 'Sci-kit Learn':
  st.header('Choice of algorithm and parameters')
  
  algo = st.radio(
          "Algorithm:",
          ('MLP', 'Perceptron'))

  with st.form(key='my_form')  :
             
      col1, col2, col3 = st.columns([1, 2, 3])
      
      with col1:
            if algo == 'MLP':
                solver = st.radio(
                    "Solver",
                    ('lbfgs', 'sgd', 'adam'))
      with col2:
            if algo == 'MLP':       
                neur = st.number_input('hidden layer size', min_value=1,
                         max_value=100, value=25, step=1)
      submit_button = st.form_submit_button(label='Submit')
      

if algo == 'MLP':
    model = MLPClassifier(random_state=44, hidden_layer_sizes=(neur,), solver=solver).fit(X_train, y_train)
elif algo == 'Perceptron':
    model = Perceptron(random_state=44).fit(X_train, y_train)
    
st.write("model score:")                                      
st.write(model.score(X_test, y_test))

#ConfusionMatrixDisplay.from_estimator(model X_test, y_test)

user_input = st.text_input('Enter a new message:', value="Are you wild ? Meet wild data analysts from your city !", )

user_input_c = netto(user_input)
user_input_c = [user_input_c]


user_input_v = vec.transform(user_input_c)   

st.write(model.predict(user_input_v))
    

       

st.sidebar.title(' ')
st.sidebar.info('A **wild** SPAM detector powered by Python and Streamlit')
