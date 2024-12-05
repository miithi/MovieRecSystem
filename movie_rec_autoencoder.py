#%%
#Movie Recommendation System using Autoencoders
#This script compresses the BoW vectors using an autoencoder and uses cosine similarity 
#on the compressed latent space for recommendations.
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Load preprocessed dataset (using the same preprocessing as in BoW)
from movie_rec_system_bow import movies_dataFrame  # reusing preprocessing

# Text vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_dataFrame['tags']).toarray()

# Define Autoencoder model
input_dim = vectors.shape[1]  # Number of features in BoW
encoding_dim = 128  # Latent space size

# Autoencoder architecture
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)

# Encoder model (extract only the encoded part)
encoder_model = tf.keras.Model(inputs=input_layer, outputs=encoded)

# Compile and train the Autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(vectors, vectors, epochs=10, batch_size=64, shuffle=True)

# Extract compressed embeddings
compressed_vectors = encoder_model.predict(vectors)


# Compute cosine similarity on latent space
similarity_cos = cosine_similarity(compressed_vectors)
similarity_euc = euclidean_distances(compressed_vectors)

def recommendCosine(movie):
    if movie not in movies_dataFrame['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    
    movie_index = movies_dataFrame[movies_dataFrame['title'] == movie].index[0]
    distances = similarity_cos[movie_index]
    recommended_movies = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in recommended_movies:
        print(movies_dataFrame.iloc[i[0]].title)

def recommendEuclidean(movie):
    if movie not in movies_dataFrame['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    
    movie_index = movies_dataFrame[movies_dataFrame['title'] == movie].index[0]
    distances = similarity_euc[movie_index]
    recommended_movies = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in recommended_movies:
        print(movies_dataFrame.iloc[i[0]].title)


print("Top recommendations based on cosine similarity:", similarity_cos[0][:5])
print("Top recommendations based on Euclidean distance:", similarity_euc[0][:5])

test_movies = ["Avatar"]
for movie in test_movies:
    print("\n" + "Recommendations based on: " + movie + "\n")
    recommendCosine(movie)
    print("\n" + "=" * 50 + "\n")
    recommendEuclidean(movie)
    print("\n" + "=" * 50 + "\n")

# %%
# Epoch: Represents how many times the model has seen the entire dataset.
# Loss: Measures how far the predicted output is from the actual input, but it's not the same as accuracy.
# In an autoencoder, you're concerned with minimizing reconstruction loss, not maximizing accuracy.