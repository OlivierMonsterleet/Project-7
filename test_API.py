import pytest

import pandas as pd
import pytest
import pickle
from flask import request
import requests
import json

import P7_API

from P7_API import app

# +
#import sys 
#sys.path

# +
#
# -

## Création d'un client
client = app.test_client()


# +
# #client?
# -

def test_get_all_data_json():
    response = client.get('/get_all_data_json')
    data = json.loads(response.text)
    df = pd.DataFrame.from_dict(data)
    assert response.status_code == 200


def test_get_client_prediction():
    response = client.get('/get_client_prediction?cid=193065')
    data = json.loads(response.text)
    assert data == 0.0
    
def test_get_client_prediction2():
    response = client.get('/get_client_prediction?cid=193065')
    data = json.loads(response.text)
    assert response.status_code == 200

def test_get_client_predict_proba():
    response = client.get('/get_client_predict_proba?cid=193065')
    data = json.loads(response.text)
    assert response.status_code == 200
    
#def test_get_client_shap():
#    response = client.get('/get_client_shap?cid=370048')
#    data = json.loads(response.text)
#    assert response.status_code == 200