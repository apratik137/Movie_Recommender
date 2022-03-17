import surprise
from surprise import SVD
from surprise import SVDpp
from surprise import SlopeOne
from surprise import CoClustering
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise  import Reader
import pandas as pd
from scipy import sparse
import numpy as np
import pickle
import urllib.request
import json
import sqlite3
import argparse
from sklearn.neighbors import NearestNeighbors



def getMovieDataJSON(user_id):
    base_url = "http://128.2.204.215:8080/user/"
    url = base_url + str(user_id)
    response = urllib.request.urlopen(url)
    return json.load(response)



def train_model(df_path):
    df = pd.read_csv(df_path)
    list_movies = pd.unique(df['movieid'])
    list_users = pd.unique(df['userid'])
    np.save('movie_list.npy', list_movies)
    np.save('user_list.npy', list_users)
    reader = Reader()
    data = Dataset.load_from_df(df[['userid', 'movieid', 'latest_rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVDpp()
    algo.fit(trainset)
    pickle.dump(algo,open('model_SVD.pkl','wb'))

def prediction(userid):
    loaded_model = pickle.load(open('model_SVD.pkl', 'rb'))
    movie_list = np.load('movie_list.npy', allow_pickle = True)
    user_list = np.load('user_list.npy', allow_pickle = True)
    # df = df = pd.read_csv('../new_data.csv')
    if userid not in user_list:
        movies = random_20()
        return movies
    
    top_ratings = []
    for i in movie_list:
        # print(i)
        # print("check")
        top_ratings.append((i,loaded_model.predict(userid,i)))
        # break
    top_ratings.sort(key = lambda x: x[1][3])
    top_ratings = top_ratings[-20:][::-1]
    return_val = ""
    for i in top_ratings:
        return_val+=f"{i[1][1]},"
    return return_val[:-1]


def random_20():
    top_100 = np.load('top_100.npy', allow_pickle = True)
    np.random.shuffle(top_100)
    ans = ""
    for i in range(20):
        ans+=f"{top_100[i]},"
    return ans[:-1]


# def top_100(data_path):
#     df =  pd.read_csv(data_path)
#     df = df.drop(columns = ['userid','num_minutes'])
#     print(df.columns)
#     print(len(df))
#     print(len(pd.unique(df['movieid'])))
#     df = df.groupby(['movieid']).mean()
#     print(len(df))
#     df  = df.sort_values(by=['latest_rating'], ascending=False)
#     df = df.head(100)
#     temp = df.index.to_numpy()
#     np.save('top_100.npy',temp)
    
    # temp = df['movieid'].to_numpy()
    # print(type(temp))

# def similar_user(userid):
#     base_url = "http://128.2.204.215:8080/user/"
#     url = base_url + str(userid)
#     response = urllib.request.urlopen(url)
#     response = json.load(response)
#     return response

# def train_similar_user(data_path):
#     df =  pd.read_csv(data_path)
#     list_users = pd.unique(df['userid'])
#     list_movies = pd.unique(df['userid'])
#     np.save('movie_list.npy', list_movies)
#     np.save('user_list.npy', list_movies)
#     temp = df.to_numpy()
#     print(temp[0:10])


# top_100('../new_data.csv')
# train_model('../new_data.csv')

temp = np.load('user_list.npy', allow_pickle = True)
print(prediction(temp[0]))
print('\n\n\n\nhello')
print(prediction(1))