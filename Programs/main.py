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
foofile = zipfile.open('ml-100k/u.data')
user_data = pd.read_csv(foofile, compression='gzip', sep='\\t')
#%%

