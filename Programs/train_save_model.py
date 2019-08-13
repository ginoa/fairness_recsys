import os
import pandas as pd

from sklearn import preprocessing

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

os.chdir("C:\\Users\\errat\\Documents\\GitHub\\fairness_recsys\\Programs")
import model

# load training data

resp = urlopen("http://files.grouplens.org/datasets/movielens/ml-100k.zip")
zipfile = ZipFile(BytesIO(resp.read()))
foofile = zipfile.open('ml-100k/u1.base')
df = pd.read_csv(foofile, usecols = [0,1,2], header = None, names = ['user_id','item_id','rating'], sep='\t')

num_items = df.item_id.nunique()
num_users = df.user_id.nunique()
print("USERS: {} ITEMS: {}".format(num_users, num_items))

#%% Process data

# Normalize in [0, 1]

r = df['rating'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(r.reshape(-1,1))
df_normalized = pd.DataFrame(x_scaled)
df['rating'] = df_normalized


# Convert DataFrame in user-item matrix

matrix = df.pivot(index='user_id', columns='item_id', values='rating')
matrix.fillna(0, inplace=True)


# Users and items ordered as they are in matrix

users = matrix.index.tolist()
items = matrix.columns.tolist()

matrix = matrix.values

print("Matrix shape: {}".format(matrix.shape))

# num_users = matrix.shape[0]
# num_items = matrix.shape[1]
# print("USERS: {} ITEMS: {}".format(num_users, num_items))

#%% Define and train model
mymodel = model.Autoencoder(input_size=len(items),hidden_layer_size=100)
mymodel.fit(X=matrix,epochs=200)
