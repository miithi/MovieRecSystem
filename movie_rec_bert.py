#%%
# Movie Recommendation System using BERT
# This script generates semantic embeddings for movie tags using BERT
# and uses cosine similarity for recommendations.


import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed dataset (use the same preprocessing as in BoW)
from movie_rec_system_bow import movies_dataFrame, recommend  # reuse preprocessing

# Generate embeddings using BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(movies_dataFrame['tags'].tolist())

# Compute cosine similarity on BERT embeddings
similarity = cosine_similarity(embeddings)

# Recommendation function

#recommend("Avatar")

# %%
