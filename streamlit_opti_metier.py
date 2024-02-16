import streamlit as st
import pickle
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json



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

                        
#model= pickle.load(open('model_opti_metier.pkl','rb'))


st.image('Pret_a_depenser_logo.png', caption='Prêt à dépenser')

        
##### MENU DEROULANT ##############
liste_clients = data['SK_ID_CURR']
col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
    cid_input = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*", 
                                (liste_clients))
    st.write("Vous avez sélectionné l'identifiant n° :", cid_input)
with col2:
        st.write("")


        

data_filtered = data.loc[data['SK_ID_CURR']==cid_input]        
data_light = data_filtered.drop(columns=['SK_ID_CURR','TARGET']) 


########## PREDICT PROBA #######################################
url = 'https://p7-api-web-service-z5hp.onrender.com/get_client_predict_proba'
pred=requests.get(url, params = {'cid': cid_input})
st.write("La proba du client selectionné est (proba/1-proba) :")
st.write(pred.text)




        
######## affichage du client sélectionné #######################
client_df = data.loc[data['SK_ID_CURR']==cid_input]



######### affichage de toutes les données ######################
st.write("Données du client VS données globales")
transpose1 = data.mean().to_frame().T               
transpose_inv = transpose1.mul(-1)


transpose_inv['scope'] = 'all'
transpose1['scope'] = 'client'
merged_data = pd.concat([client_df,transpose_inv])
merged_data = merged_data.set_index('scope')
st.dataframe(merged_data)


######### BAR CHARTS ###########################################
st.write("BAR charts des données du client VS globales")
st.bar_chart(data=merged_data.T)


################# DISTRIBUTION ################################
fig, ax = plt.subplots()
ax.hist(client_df, bins=20)
plt.title('Distribution des données du client')
plt.xlabel("DATA")
plt.ylabel("Volume")
st.pyplot(fig)


fig, ax = plt.subplots()
ax.hist(data['EXT_SOURCE_1'], bins=20)
plt.title('Distribution des données du client')
plt.xlabel("EXT_SOURCE_1")
plt.ylabel("Volume")
st.pyplot(fig)