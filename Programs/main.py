# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:43:02 2019

@authors: Gino Almondo, Elena Beretta
"""
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd


#%% load data

resp = urlopen("http://files.grouplens.org/datasets/movielens/ml-100k.zip")
zipfile = ZipFile(BytesIO(resp.read()))
#%%
foofile = zipfile.open('ml-100k/u.data')
data_ratings = pd.read_csv(foofile, usecols = [0,1,2], header = None, names = ['user_id','item_id','rating'], sep='\t')

foofile = zipfile.open('ml-100k/u.user')
data_users = pd.read_csv(foofile, header = None, names = ['user_id','user_age','user_gender','user_occupation','user_zipcode'], sep='|')

