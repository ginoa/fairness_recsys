import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from sklearn.metrics import precision_score

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

#%%

# Define evaluation metrics

eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)


# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

with tf.Session() as session:
    session.run(init)
    session.run(local_init)

    num_batches = int(matrix.shape[0] / batch_size)
    matrix = np.array_split(matrix, num_batches)

    for i in range(epochs):

        avg_cost = 0

        for batch in matrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l

        avg_cost /= num_batches

        print("Epoch: {} Loss: {}".format(i + 1, avg_cost))

        # if i % display_step == 0 or i == 1:
        #     print('Step %i: Minibatch Loss: %f' % (i, l))

    print("Predictions...")

    matrix = np.concatenate(matrix, axis=0)

    preds = session.run(decoder_op, feed_dict={X: matrix})

    # print(matrix)
    # print(preds)

    predictions = predictions.append(pd.DataFrame(preds))

    predictions = predictions.stack().reset_index(name='rating')
    predictions.columns = ['user', 'item', 'rating']
    predictions['user'] = predictions['user'].map(lambda value: users[value])
    predictions['item'] = predictions['item'].map(lambda value: items[value])

    # print(predictions)

    print("Filtering out items in training set")

    keys = ['user', 'item']
    i1 = predictions.set_index(keys).index
    i2 = df.set_index(keys).index

    recs = predictions[~i1.isin(i2)]
    recs = recs.sort_values(['user', 'rating'], ascending=[True, False])
    recs = recs.groupby('user').head(k)
    recs.to_csv('recs.tsv', sep='\t', index=False, header=False)

    # creare un vettore dove ci sono per ogni utente i suoi 10 movies

    test = pd.read_csv(test_data, sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None)
    test = test.drop('timestamp', axis=1)

    test = test.sort_values(['user', 'rating'], ascending=[True, False])

    #test = test.groupby('user').head(k) #.reset_index(drop=True)
    #test_list = test.as_matrix(columns=['item']).reshape((-1))
    #recs_list = recs.groupby('user').head(k).as_matrix(columns=['item']).reshape((-1))

    print("Evaluating...")

    p = 0.0
    for user in users[:10]:
        test_list = test[(test.user == user)].head(k).as_matrix(columns=['item']).flatten()
        recs_list = recs[(recs.user == user)].head(k).as_matrix(columns=['item']).flatten()

        session.run(pre_op, feed_dict={eval_x: test_list, eval_y: recs_list})

        #pu = precision_score(test_list, recs_list, average='micro')
        #p += pu

        # print("Precision for user {}: {}".format(user, pu))
        # print("User test: {}".format(test_list))
        # print("User recs: {}".format(recs_list))

    #p /= len(users)

    p = session.run(pre)
    print("Precision@{}: {}".format(k, p))

    # print("test len: {} - recs len: {}".format(len(test_list), len(recs_list)))
    #
    # print("test list - type: {}".format(type(test_list)))
    # print(test_list)
    #
    # print("recs list - type: {}".format(type(recs_list)))
    # print(recs_list)