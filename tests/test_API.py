#!/usr/bin/env python
# coding: utf-8

# In[1]:



from ..P7_API import get_all_data
from ..P7_API import get_all_data_json 
from ..P7_API import get_client_data
from ..P7_API import get_client_prediction
from ..P7_API import get_client_predict_proba


import pandas as pd
import pytest
import pickle

data = pickle.load(open('data_opti_metier.pkl','rb'))


def test_client_details(cid=370048):
    data = get_client_data()
    print(f'id test = {cid}')
    print(data['AMT_ANNUITY'])
    assert data['AMT_ANNUITY']==49500.0
    
    
#def test_target_col(data):
#    """Test that the train dataframe has a 'target' column"""
#    assert 'TARGET' in data.columns


