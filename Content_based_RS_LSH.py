#!/usr/bin/env python

import collections
import itertools
import math
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sys
from sklearn.metrics import mean_squared_error
import time

# returns watched movies by users and their corresponding ratings
def get_user_movies_dict(ratings_train_filename):
    
    user_movies_dict = dict()
    user_movies_ratings = pd.read_csv(ratings_train_filename)

    for index, user_movies_rating in user_movies_ratings.iterrows():
        userId = user_movies_rating['userId']
        movieId = user_movies_rating['movieId']
        rating = user_movies_rating['rating']
        
        if userId not in user_movies_dict:
            user_movies_dict[userId] = dict()
        user_movies_dict[userId][movieId] = rating
        
    return user_movies_dict

# returns genre index dictionary
def get_genre_vocab(movies_tokenized):
    doclist = movies_tokenized['tokens'].tolist()     
    vocab_set = set(i for s in doclist for i in s)
    vocab = {i: x for x, i in enumerate(sorted(list(vocab_set)))}
    return vocab

def tokenize(movies):
    tokenlist=[]
    for index,row in movies.iterrows():
        tokenlist.append(tokenize_string(row.genres))
        
    movies['tokens']=tokenlist
    return movies

def tokenize_string(my_string):
    return my_string.split('|');

def get_shingle_hash(hash_size, movies_genres):
    
    shingles = [] #list of set

    for movies_genre in movies_genres:
        tokens = list(movies_genre)

        shingle = set()
        for ngram in [[token] for token in tokens]:
            #frozenset to make it order independent and hashable
            ngram = frozenset(ngram)
            ngram_hash = hash(ngram) % hash_size

            shingle.add(ngram_hash)
        shingles.append(shingle)

    return shingles

def tokenize_genre(movies_tokenized, genre_vocab):
    
    tokenlist=[]
    for index,row in movies_tokenized.iterrows():
        tokens = set(map(lambda x: genre_vocab[x] , row.tokens) )
        tokenlist.append(tokens)        
    movies_tokenized['genre_tokens']=tokenlist
    
    return movies_tokenized    

def get_hash_coeffs(br):
    
    rnds = np.random.choice(2**10, (2, br), replace=False)
    c = 1048583
    return rnds[0], rnds[1], c

def min_hashing(shingles, hash_coeffs, br):
    count = len(shingles)

    (a, b, c) = hash_coeffs
    a = a.reshape(1, -1)
    M = np.zeros((br, count), dtype=int) 
    for i, s in enumerate(shingles):
        row_idx = np.asarray(list(s)).reshape(-1, 1)
        m = (np.matmul(row_idx, a) + b) % c
        m_min = np.min(m, axis=0) #For each hash function, minimum hash value for all shingles
        M[:, i] = m_min

    return M

def LSH(M, b, r, band_hash_size):
    count = M.shape[1]

    bucket_list = []
    for band_index in range(b):
        
        m = collections.defaultdict(set)

        row_start = band_index * r
        for c in range(count):
            v = M[row_start:(row_start+r), c]
            v_hash = hash(tuple(v.tolist())) % band_hash_size
            m[v_hash].add(c)

        bucket_list.append(m)

    return bucket_list

def find_similiar(shingles, query_index, threshold, bucket_list, M, b, r, band_hash_size, verify_by_signature):
    
    # Step 1: Find candidates
    candidates = set()
    for band_index in range(b):
        row_start = band_index * r
        v = M[row_start:(row_start+r), query_index]
        v_hash = hash(tuple(v.tolist())) % band_hash_size

        m = bucket_list[band_index]
        bucket = m[v_hash]
        candidates = candidates.union(bucket)

    # Step 2: Verify similarity of candidates
    sims = []
    
    #Since the candidates size is small, we just evaluate it on k-shingles matrix, or signature matrix for greater efficiency
    if verify_by_signature:
        query_vec = M[:, query_index]
        for col_idx in candidates:
            col = M[:, col_idx]
            sim = np.mean(col == query_vec) # Jaccard Similarity is proportional to the fraction of the minhashing signature they agree
            if sim >= threshold:
                sims.append((col_idx, sim))
    else:
        query_set = shingles[query_index]
        for col_idx in candidates:
            col_set = shingles[col_idx]

            sim = len(query_set & col_set) / len(query_set | col_set) # Jaccard Similarity
            if sim >= threshold:
                sims.append((col_idx, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims

def get_movieId_index_dicts(movies):
    
    movieId_index_dict = dict()
    index_movieId_dict = dict()
    
    for index, movie in movies.iterrows():
        movieId = movie['movieId']
        index_movieId_dict[index] = movieId
        movieId_index_dict[movieId] = index 
    
    return movieId_index_dict, index_movieId_dict

def get_predicted_ratings(user_movies_dict, ratings_test_filename, movies, test_df):
    
    b = 125
    r = 4
    threshold = (1/b)**(1/r)
    
    predicted_ratings = []
    
    movieId_index_dict, index_movieId_dict = get_movieId_index_dicts(movies)
    
    movies_genres = movies_tokenized_genre['genre_tokens'].tolist()
    
    hash_size = 2**20
    band_hash_size = 2**16
    verify_by_signature = False

    br = b * r
    
    hash_coeffs = get_hash_coeffs(br)

    shingles = get_shingle_hash( hash_size, movies_genres)
    M = min_hashing(shingles, hash_coeffs, br)
    
    bucket_list = LSH(M, b, r, band_hash_size)

    similar_movies_dict = dict()

    for index, query in test_df.iterrows():
        
        sim_rating_total = 0
        sim_total = 0
        
        query_userId = query['userId']
        watched_movies = user_movies_dict[query_userId]
        
        query_movieId = query['movieId']
        
        if query_movieId not in similar_movies_dict:
            query_index = movieId_index_dict[query_movieId]
            similar_movies_dict[query_movieId] = find_similiar(movies_genres, query_index, threshold, bucket_list, M, b, r, band_hash_size, verify_by_signature)
        
        similar_movies = similar_movies_dict[query_movieId]
        
        for similar_movie_index, similarity in similar_movies:
            
            similar_movieId = index_movieId_dict[similar_movie_index]
            
            if similar_movieId in watched_movies:
                sim_total += similarity
                sim_rating_total += similarity*watched_movies[similar_movieId]
        
        if sim_total!=0:
            #weighted mean logic
            predicted_ratings.append(sim_rating_total/sim_total)
            
        else:
            #mean logic
            #print('has no overlap')
            if len(watched_movies)!=0:
                predicted_ratings.append(sum(watched_movies.values())/len(watched_movies))
            else:
                predicted_ratings.append(3.0)
            
    return predicted_ratings    

if __name__ == '__main__':

    movies_filename = sys.argv[1]
    ratings_train_filename = sys.argv[2]
    ratings_test_filename = sys.argv[3]
    output_filename = sys.argv[4]
    
    #movies_filename = 'ml-latest-small/movies.csv'
    #ratings_train_filename = 'ml-latest-small/ratings_train.csv'
    #ratings_test_filename = 'ml-latest-small/ratings_test.csv'
    #output_filename = "task4_output"
    
    movies = pd.read_csv(movies_filename)
    movies_tokenized = tokenize(movies)
    genre_vocab = get_genre_vocab(movies_tokenized)
    movies_tokenized_genre = tokenize_genre(movies_tokenized, genre_vocab)

    user_movies_dict = get_user_movies_dict(ratings_train_filename)

    test_df = pd.read_csv(ratings_test_filename)
    
    ratings = get_predicted_ratings(user_movies_dict, ratings_test_filename, movies_tokenized_genre, test_df)
    ratings = list(map(lambda r: 0.5 if r<0.5 else r, ratings))
    ratings = list(map(lambda r: 5.0 if r>5.0 else r, ratings))
    
    test_df['rating'] = ratings
    
    test_df.to_csv(output_filename)