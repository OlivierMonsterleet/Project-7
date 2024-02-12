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
    

def test_source1():
    data=get_all_data_json()
    data = json.loads(data)
    df = pd.DataFrame.from_dict(data)
    assert 'EXT_SOURCE_1' in df.columns
    
#def test_prediction(cid=370048):
#    data = get_client_prediction(cid)
#    pred = model.predict(data_filtered)
#    assert pred==0.0
    
    
#def test_get_client_data(cid=370048):
#    data = get_client_data(cid)
#    assert data_filtered['AMT_ANNUITY']==49500
    