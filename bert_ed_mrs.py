#%%
# Movie Recommendation System using BERT
# This script generates semantic embeddings for movie tags using BERT
# and uses cosine similarity for recommendations.


import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity


# Load preprocessed dataset (use the same preprocessing as in BoW)
from movie_rec_system_bow import movies_dataFrame # reuse preprocessing

# Generate embeddings using BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(movies_dataFrame['tags'].tolist())

# Compute cosine similarity on BERT embeddings
similarity_cosine = cosine_similarity(embeddings)
similarity_euclidean = euclidean_distances(embeddings)

def recommendCosine(movie):
    if movie not in movies_dataFrame['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    
    movie_index = movies_dataFrame[movies_dataFrame['title'] == movie].index[0]
    distances = similarity_cosine[movie_index]
    recommended_movies = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in recommended_movies:
        print(movies_dataFrame.iloc[i[0]].title)

def recommendEuclidean(movie):
    if movie not in movies_dataFrame['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    
    movie_index = movies_dataFrame[movies_dataFrame['title'] == movie].index[0]
    distances = similarity_euclidean[movie_index]
    recommended_movies = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in recommended_movies:
        print(movies_dataFrame.iloc[i[0]].title)


# Compare similarity scores for different metrics
print("Top recommendations based on cosine similarity:", similarity_cosine[0][:5])
print("Top recommendations based on Euclidean distance:", similarity_euclidean[0][:5])


#testing
test_movies = ["Avatar"]
for movie in test_movies:
    print("\n" + "Recommendations based on: " + movie + "\n")
    recommendCosine(movie)
    print("\n" + "=" * 50 + "\n")
    recommendEuclidean(movie)
    print("\n" + "=" * 50 + "\n")


# %%
