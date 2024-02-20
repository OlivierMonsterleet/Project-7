import streamlit as st
import pickle
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image


############################################

# A lancer dans le shell : streamlit run streamlit_opti_metier.py

############################################



# Google cloud: https://storage.cloud.google.com/bucket_openclassrooms/P7_data.csv
# Load Dataframe
# path_df = 'https://storage.cloud.google.com/bucket_openclassrooms/P7-data.csv'
# dataframe = pd.read_csv(path_df)


################ CHARGEMENT DATA #######################
url = 'https://p7-api-web-service-z5hp.onrender.com/get_all_data_json'

data = requests.get(url)
data = json.loads(data.text)
data = pd.DataFrame.from_dict(data)


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
#st.image('Pret_a_depenser_logo.png', caption='Prêt à dépenser')

        
##### MENU DEROULANT ##############
liste_clients = data['SK_ID_CURR']
col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
    cid_input = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*", 
                                (liste_clients))
    st.write("Vous avez sélectionne l'identifiant n° :", cid_input)
with col2:
        st.write("")    

data_filtered = data.loc[data['SK_ID_CURR']==cid_input]        
data_light = data_filtered.drop(columns=['SK_ID_CURR','TARGET']) 


########## PREDICT PROBA #######################################
url = 'https://p7-api-web-service-z5hp.onrender.com/get_client_predict_proba'


response=requests.get(url, params = {'cid': cid_input})
response = response.text
response = json.loads(response) ## Ne fonctionne pas sur streamlit déployé
pred_decision = response[1]
pred_proba = response[0]
st.button("Décision finale : "+pred_decision, type="secondary")
st.button("Probabilité du client sélectionné (proba / 1-proba): "+pred_proba, type="secondary")



#pred=requests.get(url, params = {'cid': cid_input})
#st.write("La proba du client selectionné est (proba/1-proba) :")
#st.write(pred.text)



client_df = data.loc[data['SK_ID_CURR']==cid_input]


######### Données du client VS données globales ######################
st.markdown("""
            <h1 style="color:#03224c;font-size:1.9em;font-style:italic;font-weight:700;margin:0px;">
            Données du client comparées aux données globales</h1>
            """, unsafe_allow_html=True)
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

#st.write("Données du client VS données globales")

#transpose1 = data.mean().to_frame().T               
#transpose_inv = transpose1.mul(-1)
#transpose_inv['scope'] = 'all'
#transpose1['scope'] = 'client'
#merged_data = pd.concat([client_df,transpose_inv])
#merged_data = merged_data.set_index('scope')
#st.dataframe(merged_data)


######### BAR CHARTS ###########################################
st.write("")
st.markdown("""
            <h1 style="color:#03224c;font-size:1.9em;font-style:italic;font-weight:700;margin:0px;">
            BAR charts des données du client VS globales</h1>
            """, unsafe_allow_html=True)
#st.write("BAR charts des données du client VS globales")
st.bar_chart(data=merged_data.T)


################# DISTRIBUTION ################################
liste_variables = list(data_light.columns)
colonne_input = st.selectbox("Selectionnez une variable du client à etudier ", (liste_variables))

fig, ax = plt.subplots()
ax.hist(data[colonne_input], bins=20)
plt.title('Distribution des données du client')
plt.xlabel("Variable sélectionnée")
plt.ylabel("Distribution")
st.pyplot(fig)

#fig, ax = plt.subplots()
#ax.hist(client_df, bins=20)
#plt.title('Distribution des données du client')
#plt.xlabel("DATA")
#plt.ylabel("Volume")
#st.pyplot(fig)