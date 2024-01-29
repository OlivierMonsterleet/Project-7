import streamlit as st
import pandas
import pickle
import requests
from io import BytesIO
import plost

# streamlit run streamlit_test.py

url = 'http://127.0.0.1:8050/' # port avant=12993

# Google cloud: https://storage.cloud.google.com/bucket_openclassrooms/P7_data.csv


#Load Dataframe
#path_df = 'https://storage.cloud.google.com/bucket_openclassrooms/P7-data.csv'
#dataframe = pd.read_csv(path_df)

data_pickle = pickle.load(open('mydata.pkl','rb'))
model = pickle.load(open('mymodel.pkl','rb'))

data = data_pickle[['SK_ID_CURR','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','PAYMENT_RATE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_EMPLOYED_PERC','DAYS_REGISTRATION','DAYS_ID_PUBLISH','AMT_ANNUITY','ANNUITY_INCOME_PERC','REGION_POPULATION_RELATIVE','DAYS_LAST_PHONE_CHANGE']]

data_pickle_light = data_pickle.drop(columns=['index','TARGET'])


# st.line_chart(data)
#client_data = requests.get(url+'get_client_data/100002')  # puis chercher le param dans zone de text de streamlit  "text area"
#client_data = client_data.content# on passe au rslt, le dataframe 
# au lien de content:  .json()
#----------------------------------------------------------------------------
#client_df = pandas.read_json(BytesIO(client_data))
#client_df = pandas.read_json(BytesIO(client_data))

client_df = data.loc[data['SK_ID_CURR']==102337]

#client_df = client_df1.mul(-1)
st.write(client_df)
client_df['scope'] = 'client'
#client_data = pandas.DataFrame.from_dict(client_data,orient='index')



#st.write(type(df))      
#all_data = requests.get(url+'get_all_data/')  
#all_data = all_data.content
#all_df = pandas.read_csv(BytesIO(all_data))
transpose1 = data.mean().to_frame().T                # T= transposé, moyenne de tous les clients
transpose_inv = transpose1.mul(-1)

transpose_inv['scope'] = 'all'
merged_data = pandas.concat([client_df,transpose_inv])
merged_data = merged_data.set_index('scope')
st.dataframe(merged_data)

# BAR CHARTS------------------------------------
st.bar_chart(
    data=merged_data.T)
    #bar='colonnes',
    #value=['client', 'all'],
    #group=True)

    

# PREDICT PROBA 102337-----------------------------------
cid = 102337 
data_filtered = data_pickle_light.loc[data['SK_ID_CURR']==cid]
pred = model.predict_proba(data_filtered)
st.write("La proba du client 102337 est (proba/1-proba) :")
st.write(pred)



# INPUT DU CLIENT ID----------------------------------
cid_input = st.text_input('Entrez le client ID:', 'xxxxxx')
client_df_input = data.loc[data['SK_ID_CURR']==cid_input]
st.write(client_df_input)

data_filtered_input = data_pickle_light.loc[data['SK_ID_CURR']==cid_input]
#st.write(data_filtered_input)
#pred_input = model.predict_proba(data_filtered_input)

st.write("La proba du client IS saisi est : (proba/1-proba)")
#st.write(pred_input)

#client_df = data.loc[data['SK_ID_CURR']==cid_input]
#st.write(client_df)


# BUTTON ------------
st.button("Reset", type="primary")
if st.button('Say hello'):
    st.write(client_df)
else:
    st.write('bye bye')
