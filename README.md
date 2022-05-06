# SMS SPAM Detector

Developpement de modèles de Deep Learning pour pouvoir classifier des SMS en __langue anglaise__ comme était des SPAM ou non.

## Sommaire

* [Origine du projet](#origine-du-projet)
* [Screenshots](#interface)
* [Technologies](#technologies)
* [Bases de Données](#bases-de-données)
* [La Team](#la-team)

## Origine du projet

La **WebApp** crée se divise en 3 sections : 
- Une Interface graphique pour paramétrer facilement un réseau de neurones (Scikitlearn ou Tensorflow)
- La selection de 4 modèles préentrainés ou préconfigurés avec leurs performances respectives sur le dataset de test. 
- Un module permettant à l'utilisateur de rentrer un text en anglais et le classifier comme SPAM ou HAM.

## Interface

Ces analyses ont été mise à disposition au travers d’une __WebApp__ créée au travers de la plateforme __Streamlit__.

Pour lancer l'application : 
```streamlit run SPAM_Tracker.py```

## Technologies 

Projet fait entièrement en **Python**

Utilisations des librairies suivantes : 
 - Pandas
 - Sci Kit Learn
 - Tensorflow
 - Plotly
 - Streamlit

## Bases de données 

Base de donné de 5572 SMS en langue anglaise labélisés comme SPAM ou HAM (messages légitimes)

## La Team

Le projet a été réalisé par les élèves de la **Wild Code School** : 
- Franck Loiselet
- Franck Maillet
- Michael Kohler
- Julien Roborel