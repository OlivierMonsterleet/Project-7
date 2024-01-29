import streamlit as st
import pandas
import pickle
import requests
# from io import BytesIO
# import plost


############################################

# A lancer dans le shell : streamlit run streamlit_opti_metier.py

############################################

url = 'http://127.0.0.1:8060/' # port avant=12993
# Google cloud: https://storage.cloud.google.com/bucket_openclassrooms/P7_data.csv



################ CHARGEMENT DATA #######################
#Load Dataframe
#path_df = 'https://storage.cloud.google.com/bucket_openclassrooms/P7-data.csv'
#dataframe = pd.read_csv(path_df)
data_pickle = pickle.load(open('mydata.pkl','rb'))
model = pickle.load(open('mymodel.pkl','rb'))



data = data_pickle[['SK_ID_CURR','TARGET',
                    'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
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
                     'INCOME_CREDIT_PERC','INCOME_PER_PERSON']]
                        
data_light = data_pickle.drop(columns=['index','TARGET'])


##### MENU DEROULANT ##############
liste_clients = data['SK_ID_CURR']
col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
    cid_input = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*", 
                                (liste_clients))
    st.write("Vous avez sélectionné l'identifiant n° :", cid_input)
with col2:
        st.write("")
        
        
########## PREDICT PROBA #######################################
data_filtered = data_light.loc[data['SK_ID_CURR']==cid_input]
pred = model.predict_proba(data_filtered)
st.write("La probabilité du client ",cid_input, "d'avoir une diifculté de paiement de crédit est : ")
st.write(pred)  

        
######## affichage du client sélectionné #######################
client_df = data.loc[data['SK_ID_CURR']==cid_input]
client_df


######### affichage de toutes les données ######################
transpose1 = data.mean().to_frame().T               
transpose_inv = transpose1.mul(-1)

#transpose1 = transpose1.drop(columns=['index','TARGET'])
#transpose1_inv = transpose1_inv.drop(columns=['index','TARGET'])

transpose_inv['scope'] = 'all'
transpose1['scope'] = 'client'
merged_data = pandas.concat([client_df,transpose_inv])
merged_data = merged_data.set_index('scope')
st.dataframe(merged_data)


######### BAR CHARTS ###########################################
st.bar_chart(data=merged_data.T)




