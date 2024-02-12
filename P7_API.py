#!/usr/bin/env python
# coding: utf-8

# # Développement de l'API

# In[ ]:





# In[1]:


from flask import Flask # ou fast API
import requests
import pickle
import json
import pprint
import pandas as pd
import numpy as np


# In[ ]:


#!pip install plost


# In[ ]:





# In[4]:


app = Flask(__name__)
data = pickle.load(open('mydata.pkl','rb'))
model = pickle.load(open('mymodel.pkl','rb'))
app


# In[2]:


url = 'http://127.0.0.1:8050/'
url


# In[6]:


#data = data.drop(columns=['index','TARGET'])
data


# In[ ]:


data.dtypes


# In[5]:


@app.route("/hello/")
def hello():
        return json.dumps("hello")


# # get_all_data-->dictionnary

# In[6]:


@app.route("/get_all_data/")
def get_all_data5():
    return (data.to_dict())
    #return (data.csv())
    #return (data.to_dict(orient='records'))
    #return (data.to_json())


# # get_all_data-->JSON

# In[7]:


@app.route("/get_all_data_json/")
def get_all_data_json():
    return (data.to_json())


# # get_client_data

# In[8]:


@app.route("/get_client_data/<cid>")
def get_client_data17(cid):
    cid = int(cid) # transfo en integer à forcer
    data_filtered = data.loc[data['SK_ID_CURR']==cid]

    #return (data_filtered.to_dict(orient='index'))
    return (data_filtered.to_json())


# In[ ]:





# # get_prediction(client_id)

# In[23]:


@app.route("/get_client_prediction/<cid>")
def get_client_prediction5(cid):
    cid = int(cid) # transfo en integer à forcer
    
    data_filtered = data.loc[data['SK_ID_CURR']==cid]
    data_filtered = data_filtered.drop(columns=['TARGET','SK_ID_CURR']) 
    pred = model.predict(data_filtered)
    #prediction = classifier.predict_proba(filtered_data).tolist()
    if(pred>0.65): #[0][1]
        avis="Le crédit est refusé"
    else:
        avis="Le crédit est accepté"
    print(avis)
    print(pred)
    #return {'prediction': avis, proba_rembour':round(prediction[0][0],2)}
    return {str(pred[0]), avis}  # mettre zéro le model retourne un dataframe
    


# In[10]:


@app.route("/get_client_predict_proba/<cid>")
def get_client_predict_proba2(cid):
    
    cid = int(cid) # transfo en integer à forcer
    
    data_filtered = data.loc[data['SK_ID_CURR']==cid]
    
    pred = model.predict_proba(data_filtered)
    return str(pred[0])  # mettre zéro le model retourne un dataframe


# # LANCEMENT DE L API

# In[24]:


#LANCEMENT DE L API
if __name__ == '__main__':
    print("L API a démarré !")
    app.run(debug=False, port=8050)   # avant port 12993
    
    


# In[21]:


# data.columns. values. tolist()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:




