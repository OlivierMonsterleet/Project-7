#!/usr/bin/env python
# coding: utf-8

# In[1]:

import P7_API

from P7_API import get_all_data
from P7_API import get_all_data_json 
from P7_API import get_client_data
from P7_API import get_client_prediction
from P7_API import get_client_predict_proba



import pytest
import pandas as pd
import pytest
import pickle
from flask import request
import requests
import json


data = pickle.load(open('data_opti_metier.pkl','rb'))


def test_client_details(cid=370048):
    data = get_client_data()
    print(f'id test = {cid}')
    print(data['AMT_ANNUITY'])
    assert data['AMT_ANNUITY']==49500.0
    
    
def test_target_col(data):
    """Test that the train dataframe has a 'target' column"""
    assert 'TARGET' in data.columns


def test_get_all_data():
    data=get_all_data_json()
    data = json.loads(data)
    df = pd.DataFrame.from_dict(data)
    print(df.columns.to_list())
    assert df.columns.to_list()==['SK_ID_CURR','TARGET',
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
                     'INCOME_CREDIT_PERC',
                      'INCOME_PER_PERSON']

        
def test_get_prediction(cid=101077):
    pred=get_predictions(cid)
    print (pred["prediction"])
    print (pred['proba_rembour'])
    assert pred["prediction"]=='Crédit accepté'
    assert pred['proba_rembour']==0.49        
        