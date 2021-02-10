#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import time

def get_movieId_index_dict(movies):
    unique_ids = movies['movieId'].unique()    
    movieId_index_dict = dict()
    index = 0
    for unique_id in unique_ids:
        movieId_index_dict[unique_id] = index 
        index += 1
    return movieId_index_dict

def get_userId_index_dict(users):    
    unique_ids = users['userId'].unique()
    userId_index_dict = dict()
    index = 0
    for unique_id in unique_ids:
        userId_index_dict[unique_id] = index 
        index += 1
    return userId_index_dict

def get_metadata(train_movies_df, movieId_index_dict, userId_index_dict, bias):   
    rows = len(userId_index_dict)
    cols = len(movieId_index_dict)    
    R = np.empty((rows,cols,))
    R.fill(np.nan)
    
    for index, row in train_movies_df.iterrows():
        
        user_index = userId_index_dict[row['userId']]
        movie_index = movieId_index_dict[row['movieId']]        
        R[user_index][movie_index] = row['rating']
        
    global_mean = np.nanmean(R)
    users_mean = np.nanmean(R, axis = 1)
    movies_mean = np.nanmean(R, axis = 0)
  
    row, col = R.shape

    for i_row in range(row):        
        for i_col in range(col):            
            if np.isnan(R[i_row][i_col]):               
                R[i_row][i_col] = users_mean[i_row] + movies_mean[i_col] - global_mean + bias                
            if R[i_row][i_col] > 5.0:
                R[i_row][i_col] = 5.0
            if R[i_row][i_col] < 0.5:
                R[i_row][i_col] = 0.5
                
    return R, users_mean

def svd_predict(R, users_mean, test_movies_df, movieId_index_dict, userId_index_dict, watched_movieIds, k):
    
    U, sigma, Vt =  svds(R, k)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
    
    pred_ratings = []
    
    for index, query in test_movies_df.iterrows():        
        query_userId = query['userId']        
        user_index = userId_index_dict[query_userId]
        query_movieId = query['movieId']
        
        if query_movieId in watched_movieIds:
            query_movie_index = movieId_index_dict[query_movieId]
            pred_ratings.append(predicted_ratings[user_index][query_movie_index])
        else:
            pred_ratings.append(users_mean[user_index])
    
    return pred_ratings

if __name__ == '__main__':
        
    ratings_train_filename = sys.argv[2]
    ratings_test_filename = sys.argv[3]
    output_filename = sys.argv[4]
    
    #ratings_train_filename = 'ml-latest-small/ratings_train.csv'
    #ratings_test_filename = 'ml-latest-small/ratings_test.csv'
    #output_filename = "task4_output"
    
    train_movies_df = pd.read_csv(ratings_train_filename)
    test_movies_df = pd.read_csv(ratings_test_filename)
    
    movieId_index_dict = get_movieId_index_dict(train_movies_df)    

    userId_index_dict = get_userId_index_dict(train_movies_df)
    bias = 0
    R, users_mean = get_metadata(train_movies_df, movieId_index_dict, userId_index_dict, bias)
    k = 15    

    watched_movieIds = set(train_movies_df['movieId'])
    
    ratings = svd_predict(R, users_mean, test_movies_df, movieId_index_dict, userId_index_dict, watched_movieIds, k)
    ratings = list(map(lambda r: 0.5 if r<0.5 else r, ratings))
    ratings = list(map(lambda r: 5.0 if r>5.0 else r, ratings))    
    
    test_movies_df['rating'] = ratings    
    test_movies_df.to_csv(output_filename)