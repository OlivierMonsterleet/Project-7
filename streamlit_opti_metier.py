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

url = 'https://p7-api-web-service.onrender.com/get_all_data_json'
data = requests.get(url)
data = json.loads(data.text)
data = pd.DataFrame.from_dict(data)


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

url = 'https://p7-api-web-service.onrender.com/get_client_predict_proba'
response=requests.get(url, params = {'cid': cid_input})
response = response.text
response = json.loads(response) ## Ne fonctionne pas sur streamlit déployé
pred_decision = response[1]
pred_proba = response[0]
st.button("Décision finale : "+pred_decision, type="secondary")
st.button("Score du client sélectionné : "+pred_proba, type="secondary")



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
#transpose1['scope'] = 'client'
client_df['scope'] = 'client'
merged_data = pd.concat([client_df,transpose_inv])
merged_data = merged_data.set_index('scope')
merged_data=merged_data.drop(columns=['SK_ID_CURR'])
st.dataframe(merged_data)

######### BAR CHARTS ###########################################
st.write("BAR charts des données du client VS globales")
st.bar_chart(data=merged_data.T)

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
colonne_input = st.selectbox("Sélectionnez une variable à étudier ", (liste_variables))


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax1.hist(client_df[colonne_input], bins=20)
ax2.hist(data[colonne_input], bins=20)
ax1.set_title('Données du client')
ax2.set_title('Distribution des données globales')
plt.subplots_adjust(left=0.2, wspace=0.2, top=0.85) 
st.pyplot(fig)

