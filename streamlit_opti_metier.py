import streamlit as st
import pandas as pd
import pickle
import requests
##from io import BytesIO
##import plost
import matplotlib.pyplot as plt
import sklearn
import json
import numpy as np
from PIL import Image
import streamlit.components.v1 as components
from streamlit_shap import st_shap 
import shap
from flask import Flask, request


st.set_page_config(
        page_title='Projet 7',
        page_icon = "📊",
        layout="wide" )
# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('Pret_a_depenser_logo.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")

# +
#st.image('logo_projet_fintech.png',width = 150)
# -

##### CHARGEMENT DES DONNEES ##############

url = 'https://p7-api-web-service.onrender.com/get_all_data_json'

data = requests.get(url)

data = json.loads(data.text)
data = pd.DataFrame.from_dict(data)
model = pickle.load( open('model_opti_metier.pkl','rb'))
explainer = shap.TreeExplainer(model)


##### MENU DEROULANT ##############
liste_clients = data['SK_ID_CURR']
st.markdown("""
            <h1 style="color:#03224c;font-size:1.9em;font-style:italic;font-weight:700;margin:0px;">
            Veuillez sélectionner le numéro de votre client <br /> à l'aide du menu déroulant :</h1>
            """, 
            unsafe_allow_html=True)
st.write("")
col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
    cid_input = st.selectbox("*Sélection 👇*", 
                                (liste_clients))
    st.write("Vous avez sélectionné l'identifiant n° :", cid_input)
with col2:
        st.write("")


# +
########## PREDICTION ###############
#from time import sleep
#url = 'http://127.0.0.1:5000/get_client_predict_proba'

url = 'https://p7-api-web-service.onrender.com/get_client_predict_proba'
response=requests.get(url, params = {'cid': cid_input})
response = response.text
response = json.loads(response) ## Ne fonctionne pas sur streamlit déployé
pred_decision = response[1]
pred_proba = response[0]
st.button("Décision de crédit (min. requis= 0.7): "+pred_decision+" (Score = "+pred_proba[1:5]+")", type="secondary")



######## affichage du client sélectionné ########
client_df = data.loc[data['SK_ID_CURR']==cid_input]

# +
######### affichage de toutes les données ######################
st.markdown("""
            <h1 style="color:#03224c;font-size:1.9em;font-style:italic;font-weight:700;margin:0px;">
            Données du client VS données globales</h1>
            """, 
            unsafe_allow_html=True)
st.write("")

transpose1 = data.mean().to_frame().T               
transpose_inv = transpose1.mul(-1)
# -

transpose_inv['scope'] = 'all'
client_df['scope'] = 'client'
merged_data = pd.concat([client_df,transpose_inv])
merged_data = merged_data.set_index('scope')
merged_data=merged_data.drop(columns=['SK_ID_CURR'])
st.dataframe(merged_data)

######### BAR CHARTS ###########################################
st.markdown("""
            <h1 style="color:#03224c;font-size:1.9em;font-style:italic;font-weight:700;margin:0px;">
            BAR charts des données du client / données globales</h1>
            """, 
            unsafe_allow_html=True)
st.bar_chart(data=merged_data.T)


# +
################# SHAP ################################
st.markdown("""
            <h1 style="color:#03224c;font-size:1.9em;font-style:italic;font-weight:700;margin:0px;">
            Interprétabilité des données client : </h1>
            """, 
            unsafe_allow_html=True)
st.write("")
st.write("")
plt.figure(figsize=(8,4))



st_shap(shap.bar_plot(explainer.shap_values(client_df[[
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'PAYMENT_RATE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'DAYS_EMPLOYED_PERC',
    'DAYS_REGISTRATION',
    'DAYS_ID_PUBLISH',
    'AMT_ANNUITY',
    'ANNUITY_INCOME_PERC',
    'REGION_POPULATION_RELATIVE',
    'DAYS_LAST_PHONE_CHANGE',
    'INCOME_CREDIT_PERC',
    'INCOME_PER_PERSON']].values.reshape(1,-1))[0][0],\
              feature_names=[
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'PAYMENT_RATE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'DAYS_EMPLOYED_PERC',
    'DAYS_REGISTRATION',
    'DAYS_ID_PUBLISH',
    'AMT_ANNUITY',
    'ANNUITY_INCOME_PERC',
    'REGION_POPULATION_RELATIVE',
    'DAYS_LAST_PHONE_CHANGE',
    'INCOME_CREDIT_PERC',
    'INCOME_PER_PERSON'],max_display=15))
# +
################# DISTRIBUTION ################################
st.markdown("""
            <h1 style="color:#03224c;font-size:1.9em;font-style:italic;font-weight:700;margin:0px;">
            Distribution des données</h1>
            """, 
            unsafe_allow_html=True)
st.write("")
st.write("")

data_sans_id_sk = data.drop(columns=['SK_ID_CURR'])
liste_variables = list(data_sans_id_sk.columns)
colonne_input = st.selectbox("Sélectionnez une variable pour étudier sa distribution :", (liste_variables))


ditrib_min = data[colonne_input].ravel().min()
distrib_max = data[colonne_input].ravel().max()
valeur_client = client_df[colonne_input]
fig, ax = plt.subplots(1,1, figsize=(15, 5))

ax.hist(data[colonne_input], bins=20)
plt.axline((valeur_client[0], ditrib_min), (valeur_client[0], distrib_max), c='darkorange', ls='dashed', linewidth=10, label="Client sélectionné")
labels = ["Global","horizontal"]
handles, _ = ax.get_legend_handles_labels()

plt.legend(handles = handles[1:], labels = labels)

st.pyplot(fig)