#!/usr/bin/env python
# coding: utf-8
# %%
# # Développement de l'API

# %%
from flask import Flask, jsonify
import requests
import pickle
import json
import pprint
import pandas as pd
import numpy as np
import shap
from streamlit_shap import st_shap

# %%
from flask import request

# %%
app = Flask(__name__)

# %%
data = pickle.load(open('data_opti_metier.pkl','rb'))
model = pickle.load(open('model_opti_metier.pkl','rb'))
explainer = shap.TreeExplainer(model)


# %%
@app.route('/get_all_data_json', methods=['GET'])
def get_all_data_json():
    return (data.to_json())





@app.route('/get_client_data', methods=['GET'])
def get_client_data(): 
    cid=request.args.get('cid')
    data_filtered = data.loc[data['SK_ID_CURR']==int(cid)]
    data_filtered=data_filtered.drop(columns=['SK_ID_CURR','TARGET'])
    return (data_filtered.to_json())
# %%


@app.route("/get_client_prediction", methods=['GET'])
def get_client_prediction():
    cid=request.args.get('cid')
    print (cid)
    data_filtered = data.loc[data['SK_ID_CURR']==int(cid)]
    print (data_filtered)
    data_filtered=data_filtered.drop(columns=['SK_ID_CURR','TARGET'])
    pred = model.predict(data_filtered)
    return str(pred[0])  # mettre zéro le model retourne un dataframe

    



# %%
@app.route("/get_client_predict_proba", methods=['GET'])
def get_client_predict_proba():
    
    cid=request.args.get('cid')
    data_filtered = data.loc[data['SK_ID_CURR']==int(cid)]
    data_filtered=data_filtered.drop(columns=['SK_ID_CURR','TARGET'])
    pred = model.predict_proba(data_filtered)
    
    if (pred[0][0] > 0.7020000000000001):
        avis = "Le score du client est superieur au seuil d'acceptabilite (0.7), le credit est ACCEPTE."
    else:
        avis = "Le score du client est inferieur au seuil d'acceptabilite (0.7), le credit est REFUSE."
        
    return [str(pred[0]), avis] #str(pred[0]),  # mettre zéro le model retourne un dataframe




@app.route("/get_client_shap", methods=['GET'])
def fct_shap():
    cid=request.args.get('cid')
    data_filtered = data.loc[data['SK_ID_CURR']==int(cid)]                 
    data_filtered=data_filtered.drop(columns=['SK_ID_CURR','TARGET'])
    shap_values = explainer(data_filtered)
    return (shap_values.to_json())
    #return jsonify(shap_values[0,:,1])
    


# %%
#LANCEMENT DE L API
if __name__ == '__main__':
    print("L API a démarré !")
    app.run(debug=False, host='0.0.0.0')   # avant port 12993
    #app.run(debug=False, host='https://p7-api-web-service.onrender.com/')
    ##app.run()

# %%