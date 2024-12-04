#%%
#Movie Recommendation System using Autoencoders
#This script compresses the BoW vectors using an autoencoder and uses cosine similarity 
#on the compressed latent space for recommendations.
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed dataset (using the same preprocessing as in BoW)
from movie_rec_system_bow import movies_dataFrame, recommend  # reusing preprocessing

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
similarity = cosine_similarity(compressed_vectors)

# Recommendation function
recommend("Avatar")


# %%
# Epoch: Represents how many times the model has seen the entire dataset.
# Loss: Measures how far the predicted output is from the actual input, but it's not the same as accuracy.
# In an autoencoder, you're concerned with minimizing reconstruction loss, not maximizing accuracy.