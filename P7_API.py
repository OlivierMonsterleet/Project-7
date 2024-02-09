#!/usr/bin/env python
# coding: utf-8

# # Développement de l'API

# In[ ]:


# !pip install plost


# In[20]:


# pip install pytest


# In[ ]:





# In[1]:


from flask import Flask
from flask import request
import requests
import pickle
import json
import pprint
import pandas as pd
import numpy as np


# In[3]:


app = Flask(__name__)


data = pickle.load(open('data_opti_metier.pkl','rb'))
model = pickle.load(open('model_opti_metier.pkl','rb'))

app


# In[18]:


#url = 'http://127.0.0.1:8050/'


# In[ ]:


#   http://127.0.0.1:8050/get_client_data/248265




# In[7]:


@app.route("/hello/")
def hello():
        return json.dumps("hello")


# # get_all_data en dictionnary

# In[8]:


@app.route("/get_all_data/",methods=['GET'])
def get_all_data():
    return (data.to_dict())



# In[11]:


@app.route("/get_all_data_json/",methods=['GET'])
def get_all_data_json():
    return (data.to_json())


# In[16]:


@app.route("/get_client_data",methods=['GET'])
def get_client_data():
    cid=request.args.get('cid')
    data_filtered = data.loc[data['SK_ID_CURR']==cid]
    return (data_filtered.to_json())


# In[ ]:




# In[10]:


@app.route("/get_client_prediction", methods=['GET'])
def get_client_prediction():
    cid=request.args.get('cid')
    data_filtered = data.loc[data['SK_ID_CURR']==cid]
    
    pred = model.predict(data_filtered)
    return str(pred[0])  # mettre zéro le model retourne un dataframe
    


# In[13]:

         
@app.route("/get_client_predict_proba", methods=['GET'])
def get_client_predict_proba():
    
    cid=request.args.get('cid')
    
    data_filtered = data.loc[data['SK_ID_CURR']==int(cid)]
    data_filtered=data_filtered.drop(columns=['SK_ID_CURR'])
    pred = model.predict_proba(data_filtered)
    return str(pred[0])


# In[17]:


#LANCEMENT DE L API
if __name__ == '__main__':
    print("L API a démarré !")
    app.run(debug=False, host='0.0.0.0')
    
    


# In[4]:


# In[5]:


'''data1 = data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
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
data1'''


# In[ ]:



