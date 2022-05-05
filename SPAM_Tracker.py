import numpy as np
import pandas as pd
import streamlit as st
import spacy
import pickle
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline

# CONFIG

st.set_page_config(page_title="SPAM Tracker",
                   page_icon="📲",
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


@st.cache
def load_data(url):
    '''Importe la donnée et la met en cache'''
    df = pd.read_csv(url, index_col=0)
    df.fillna('', inplace=True)
    return df


def space(n):
    '''Créé des espaces de présentation'''
    for _ in range(n):
        st.title(' ')


def clean_sms(sms):
    '''Clean SMS'''
    sent_tokens = nlp(sms.lower())
    lem_sms = [token.lemma_ for token in sent_tokens if not (token.is_punct | token.is_stop)]
    return ' '.join(lem_sms)


def binary_traduction(predictions):
    '''Transforme l'output de prédiciton Tensorflow en liste de 0 et 1.
    Tout ce qui est inférieur à un seuil est ramené à 0, le reste est égal à 1'''
    pred = []
    for el in predictions:
        if el < 0.5:       # 0,5 si sigmoid en output ou 0 si rien
            pred.append(0)
        else:
            pred.append(1)
    return pred


def vectorize_text(text):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text)


# DATA

nlp = spacy.load('en_core_web_sm')

model_tfw = tf.keras.models.load_model('/Users/miko/Documents/Dev/Github/SMS_SPAM_Detector/models/model_tensorflow')

SMS = "/Users/miko/Documents/Dev/Github/SMS_SPAM_Detector/data/SMS_collection.csv"
LEMMA = '/Users/miko/Documents/Dev/Github/SMS_SPAM_Detector/data/lemma _data.csv'
LABEL = '/Users/miko/Documents/Dev/Github/SMS_SPAM_Detector/data/label.csv'
sms_data = load_data(SMS)
lemma_data = load_data(LEMMA)
label_data = load_data(LABEL)

split_size = int(sms_data.shape[0]*.75)
train_text = lemma_data.iloc[:split_size]['spacy_lem']
train_label = label_data.iloc[:split_size]['label']
test_text = lemma_data.iloc[split_size:]['spacy_lem']
test_label = label_data.iloc[split_size:]['label']


# Tensorflow Text Vectorisor
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=800)
vectorize_layer.adapt(train_text)

# Scikitlearn Text Verctorisor
vec = CountVectorizer().fit(train_text)
train_text_cv = vec.transform(train_text)
test_text_cv = vec.transform(test_text)


# SIDE BAR

thematiques = [
    "Accueil",
    "Faites votre modèle",
    "Classer un SMS"
    ]

st.sidebar.title('SMS Tracker')
st.sidebar.title(' ')

section = st.sidebar.radio(
    'Selection de la partie : ',
    thematiques)
st.sidebar.title(' ')


# MAIN PAGE

if section == 'Accueil':

    data = pd.merge(label_data, sms_data, left_index=True, right_index=True)

    with st.sidebar.expander('Options de présentation'):
        limit_range = st.checkbox('Graphiques resserrés')
    data_study = data.copy()
    data_study['length'] = data_study['message'].apply(len)
    data_study = data_study.replace({'label':{0:'HAM', 1:'SPAM'}})

    st.title("Présentation du dataset d'entrainement")
    space(1)

    st.markdown(
        '''
        Les modèles de Neural Networks ont été entrainés sur un jeu de données labélisées de SMS.
        '''
    )
    space(1)

    ds_shape = data.groupby('label').describe()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            'Nombre de SMS',
            len(data)
            )
    with col2:
        st.metric(
            'Nombre de SPAM',
            ds_shape.iloc[1, 0]
            )
    with col3:
        st.metric(
            'Nombre de HAM',
            ds_shape.iloc[0, 0]
            )

    space(2)
    st.header('Premières distinctions entre HAM et SPAM')

    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(data=[go.Pie(labels=['HAM', 'SPAM'], values=ds_shape.iloc[:,0], hole=.6)])
        fig.update_traces(textposition='outside', textinfo='percent+label', textfont_size=18)
        fig.update_layout(
            margin=dict(l=40, r=40, b=10, t=50),
            showlegend=False,)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        max_range = 200 if limit_range else 800
        fig = px.histogram(data_study, x="length", color="label", marginal="box", hover_data=data_study.columns)
        fig.update_traces(opacity=0.75)
        fig.update_layout(
            barmode='overlay',
            margin=dict(l=10, r=10, b=10, t=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.00,
                xanchor="right",
                x=1),
            xaxis=dict(range=[0, max_range]))
        st.plotly_chart(fig, use_container_width=True)

if section == 'Faites votre modèle':

    data_study = data.copy()

    st.title("Faites votre modèle")
    st.header("Plongez dans la pronfondeur de l'IA")
    space(1)

if section == 'Classer un SMS':

    st.title("Classification des SMS")

    sub_section = st.sidebar.radio(
        'Selection de la sous-section :',
        ['Testez les modèles',
         'Pésentation des 4 Fantastiques', 
        ])
    st.sidebar.title(' ')

    if sub_section == 'Testez les modèles':
        
        st.header("HAM vs SPAM")
        st.markdown(
            '''
            Dans ce module, vous pouvez écrire un message __en anglais__ et vérifier si ce message est considéré 
            comme un SPAM ou non en fonction de l'algorithme choisi. 
            '''
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            txt = st.text_area('SMS à analyser',"Ecrivez votre texte", height=120)
            submit = st.button('Soumettre')
        with col2:
            select_model = st.radio(
                'Selection de la librairie : ',
                ['Sci-kit Learn', 'Tensorflow'])
            select_skl = None
            if select_model == 'Sci-kit Learn':
                select_skl = st.radio(
                    'Selection du type de modèle : ',
                    ['Perceptron', 'Multilayer Perceptron', 'SGD Classifier'])

        st.markdown('---')
        if submit:
            c_txt = clean_sms(txt)

            if select_model == 'Tensorflow':
                vector_txt = vectorize_text(c_txt)
                pred_txt = binary_traduction(model_tfw(vector_txt))
                result = pred_txt[0]
                predictions = binary_traduction(model_tfw(vectorize_text(test_text)).numpy())
                eval_histo = model_tfw.evaluate(vectorize_text(test_text), test_label, verbose=2)
                score = eval_histo[1]

            if select_skl == 'Perceptron':
                ppn = Perceptron(random_state=44).fit(train_text_cv, train_label)
                score = ppn.score(test_text_cv, test_label)
                c_txt_cv = vec.transform([c_txt])
                result = ppn.predict(c_txt_cv)
                predictions = ppn.predict(test_text_cv)

            if select_skl == 'Multilayer Perceptron':
                mlp = MLPClassifier(random_state=44, hidden_layer_sizes=(25,), solver='lbfgs').fit(train_text_cv, train_label)
                score = mlp.score(test_text_cv, test_label)
                c_txt_cv = vec.transform([c_txt])
                result = mlp.predict(c_txt_cv)
                predictions = mlp.predict(test_text_cv)

            if select_skl == 'SGD Classifier':
                text_clf = Pipeline([
                    ('count_vec', CountVectorizer()), 
                    ('tfidf_transformer', TfidfTransformer()),
                    ('clf', SGDClassifier( n_jobs=-1))]).fit(train_text, train_label)
                score = text_clf.score(test_text, test_label)
                result = text_clf.predict([c_txt])
                predictions = text_clf.predict(test_text)
        
            cm = tf.math.confusion_matrix(test_label, predictions, 2)
            fig = ff.create_annotated_heatmap(cm.numpy()[::-1], x=['HAM', 'SPAM'], y=['SPAM', 'HAM'], colorscale='Viridis')
            fig.update_layout(margin=dict(l=30, r=30, b=50, t=10), width=600, height=300)
            fig['data'][0]['showscale'] = True

            result_txt = 'SPAM' if result == 1 else 'HAM'
            st.header(f'Selon le modèle {select_model}, le message est un {result_txt}')
            space(1)
            col1, col2 = st.columns(2)
            with col1:
                if result == 1:
                    space(1)
                    st.image('/Users/miko/Documents/Dev/Github/SMS_SPAM_Detector/resources/spam.jpg', width=400)
                if result == 0:
                    st.image('/Users/miko/Documents/Dev/Github/SMS_SPAM_Detector/resources/ham.jpg', width=300)
            with col2:
                st.subheader('Confiance du modèle')
                st.metric(
                    'Score sur le train set : ',
                    round(score, 4)
                    )
                st.plotly_chart(fig, use_container_width=True)
        
    if sub_section == 'Pésentation des 4 Fantastiques':
        
        st.header("Les modèles les plus chauds de ta région")
        st.markdown(
            '''
            Quatres modèles ont été retenus et préentrainés pour classifier les SMS en SPAM ou HAM.
            Les librairies __Tensorflow__ et __Sci-Kit Learn__ ont été utiliés et un focus a été fait sur les _réseaux de neurones artificiels_.
            Un modèle de Machine Learning traditionnel a été retenu pour comparer les technologies. 
            '''
        )

        ppn = Perceptron(random_state=44).fit(train_text_cv, train_label)
        score_ppn = ppn.score(test_text_cv, test_label)
        predictions_ppn = ppn.predict(test_text_cv)
        cm_ppn = tf.math.confusion_matrix(test_label, predictions_ppn, 2)


        mlp = MLPClassifier(random_state=44, hidden_layer_sizes=(25,), solver='lbfgs').fit(train_text_cv, train_label)
        score_mlp = mlp.score(test_text_cv, test_label)
        predictions_mlp = mlp.predict(test_text_cv)
        cm_mlp = tf.math.confusion_matrix(test_label, predictions_mlp, 2)

        text_clf = Pipeline([
            ('count_vec', CountVectorizer()), 
            ('tfidf_transformer', TfidfTransformer()),
            ('clf', SGDClassifier( n_jobs=-1))]).fit(train_text, train_label)
        score_sgd = text_clf.score(test_text, test_label)
        predictions_sgd = text_clf.predict(test_text)
        cm_sgd = tf.math.confusion_matrix(test_label, predictions_sgd, 2)

        score_tfw = model_tfw.evaluate(vectorize_text(test_text), test_label, verbose=2)[1]
        predictions_tfw = binary_traduction(model_tfw(vectorize_text(test_text)).numpy())
        cm_tfw = tf.math.confusion_matrix(test_label, predictions_tfw, 2)

        st.subheader('Score sur le dataset de test :')
        col1, col2, col3, col4, col5, col6 = st.columns([1, 4, 3, 1, 4, 3])
        with col2:
            st.metric('Perceptron', round(score_ppn, 4))
        with col3:
            st.metric('Multilayer Perceptron', round(score_mlp, 4))
        with col5:
            st.metric('TensorFlow', round(score_tfw, 4))
        with col6:
            st.metric('SGD Classifier', round(score_sgd, 4))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fig = ff.create_annotated_heatmap(cm_ppn.numpy()[::-1], x=['HAM', 'SPAM'], y=['SPAM', 'HAM'], colorscale='Viridis')
            fig.update_layout(margin=dict(l=10, r=30, b=50, t=10), width=600, height=200)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = ff.create_annotated_heatmap(cm_mlp.numpy()[::-1], x=['HAM', 'SPAM'], y=['SPAM', 'HAM'], colorscale='Viridis')
            fig.update_layout(margin=dict(l=10, r=30, b=50, t=10), width=600, height=200)
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = ff.create_annotated_heatmap(cm_tfw.numpy()[::-1], x=['HAM', 'SPAM'], y=['SPAM', 'HAM'], colorscale='Viridis')
            fig.update_layout(margin=dict(l=10, r=30, b=50, t=10), width=600, height=200)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = ff.create_annotated_heatmap(cm_sgd.numpy()[::-1], x=['HAM', 'SPAM'], y=['SPAM', 'HAM'], colorscale='Viridis')
            fig.update_layout(margin=dict(l=10, r=30, b=50, t=10), width=600, height=200)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Vitesse d'execution :")






    space(1)

# execution du model, 
    # juste un blox de text, 
    # dessous, la selection du model,
    # A coté l'image

# Récap descriptif des models :
    # temps d'execution (training)
    # temps d'exe pour 1 prediction
    # calcul temps d'exe rapport à la précision
    # score sur le test

st.sidebar.title(' ')
st.sidebar.info('Code source disponible dans le [Depôt de données](https://github.com/MickaelKohler/SMS_SPAM_Detector) Github')