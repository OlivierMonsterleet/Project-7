#!/usr/bin/env python
# coding: utf-8

# # Développement de l'API

# In[ ]:


# !pip install plost


# In[20]:


# pip install pytest


# In[41]:


# pip install -U pytest
import pytest


# In[42]:


pytest


# In[ ]:





# In[43]:


from flask import Flask, request
import requests
import pickle
import json
import pprint
import pandas as pd
import numpy as np


# In[6]:


import pkg_resources
pkg_resources.get_distribution('flask').version


# In[63]:


app = Flask(__name__)

 
data = pickle.load(open('data_opti_metier.pkl','rb'))   #----------------> 17 colonnes !!!!!!
model = pickle.load(open('model_opti_metier.pkl','rb'))

app


# In[47]:


url = 'http://127.0.0.1:8050/'



# In[54]:





# In[ ]:





# In[ ]:





# In[69]:


@app.route("/hello/")
def hello():
        return json.dumps("hello")


# # get_all_data en dictionnary

# In[70]:


@app.route("/get_all_data/")
def get_all_data():
    return (data.to_dict())


# # get_all_data en JSON

# In[71]:


@app.route("/get_all_data_json/")
def get_all_data_json():
    return (data.to_json())


# # get_client_data

# In[72]:


@app.route("/get_client_data/<cid>")
def get_client_data(cid):
    cid = int(cid) # transfo en integer à forcer
    data_filtered = data.loc[data['SK_ID_CURR']==cid]

    return (data_filtered.to_json())


# In[ ]:





# # get_prediction(client_id ex:370048)

# In[73]:


@app.route("/get_client_prediction/<cid>")
def get_client_prediction(cid):
    cid = int(cid) # transfo en integer à forcer
    data_filtered = data.loc[data['SK_ID_CURR']==cid]
    data_filtered = data_filtered.drop(columns=['TARGET','SK_ID_CURR'])  #---> on revient aux 15 champs du modèle
    pred = model.predict(data_filtered)
    return str(pred[0])  # mettre zéro le model retourne un dataframe
    


# # get_client_predict_proba

# In[74]:


@app.route("/get_client_predict_proba/<cid>")
def get_client_predict_proba1(cid):
    
    cid = int(cid) # transfo en integer à forcer
    
    data_filtered = data.loc[data['SK_ID_CURR']==cid]
    data_filtered = data_filtered.drop(columns=['TARGET','SK_ID_CURR'])  #---> on revient aux 15 champs du modèle
    pred = model.predict_proba(data_filtered)
    return str(pred[0])  # mettre zéro le model retourne un dataframe


# # LANCEMENT DE L API

# In[75]:


if __name__ == '__main__':
    print("L API a démarré !")
    app.run(debug=False, port=8050)  
    
    


# In[6]:


data.columns. values. tolist()


# 

# In[ ]:





# In[ ]:





# In[58]:


data
# 196588


# In[68]:


cid = 370048
#data = data.drop(columns=['SK_ID_CURR'])
data_filtered = data.loc[data['SK_ID_CURR']==cid]
data_filtered = data_filtered.drop(columns=['TARGET','SK_ID_CURR'])
pred = model.predict_proba(data_filtered)
pred

