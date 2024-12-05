#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score


from movie_rec_system_bow import movies_dataFrame, similarity
from movie_rec_autoencoder import similarity_cos
from bert_ed_mrs import similarity_cosine



# Example: Compare similarity scores for the movie "Avatar"
movie_index = movies_dataFrame[movies_dataFrame['title'] == "Avatar"].index[0]

# Extract scores for the three models
bow_scores = similarity[movie_index]
bert_scores = similarity_cosine[movie_index]
autoencoder_scores = similarity_cos[movie_index]  

# Plot the scores
plt.figure(figsize=(10, 6))
plt.plot(bow_scores, label='BoW Cosine Similarity', alpha=0.7)
plt.plot(bert_scores, label='BERT Cosine Similarity', alpha=0.7)
plt.plot(autoencoder_scores, label='Autoencoder Cosine Similarity', alpha=0.7)
plt.xlabel('Movie Index')
plt.ylabel('Cosine Similarity Score')
plt.title('Cosine Similarity Comparison for "Avatar"')
plt.legend()
plt.show()


# # Assuming you have precision and recall values computed for different thresholds
# thresholds = np.linspace(0, 1, 50)
# precision_bow = [np.random.uniform(0.7, 0.9) for _ in thresholds]  # Placeholder values
# recall_bow = [np.random.uniform(0.6, 0.8) for _ in thresholds]

# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, precision_bow, label='Precision (BoW)', alpha=0.7)
# plt.plot(thresholds, recall_bow, label='Recall (BoW)', alpha=0.7)
# plt.xlabel('Threshold')
# plt.ylabel('Score')
# plt.title('Precision vs Recall for BoW')
# plt.legend()
# plt.show()

# Placeholder for true relevant movies
true_relevant = {
    "Avatar": ["Titanic", "Avengers: Endgame", "Star Wars"],
    "Titanic": ["Avatar", "The Notebook", "A Walk to Remember"],
}

def evaluate_model(model, movie_list, true_relevant, top_n=5):
    precision_scores = []
    recall_scores = []
    
    for movie in movie_list:
        if movie in true_relevant:
            # Get recommendations from the model
            recommendations = model(movie, top_n=top_n)  # Your recommend function
            
            # Ground truth and predictions
            ground_truth = set(true_relevant[movie])
            predicted = set(recommendations)
            
            # Calculate precision and recall
            tp = len(predicted & ground_truth)
            precision = tp / len(predicted) if len(predicted) > 0 else 0
            recall = tp / len(ground_truth) if len(ground_truth) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
    
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    
    return avg_precision, avg_recall

# Define test movies
test_movies = ["Avatar", "Titanic"]

# BOW Recommendation function
def recommend(movie):
    if movie not in movies_dataFrame['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    
    movie_index = movies_dataFrame[movies_dataFrame['title'] == movie].index[0]
    distances = similarity[movie_index]
    recommended_movies = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in recommended_movies:
        print(movies_dataFrame.iloc[i[0]].title)

# Bert Recommendation function
def recommendCosine(movie):
    if movie not in movies_dataFrame['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    
    movie_index = movies_dataFrame[movies_dataFrame['title'] == movie].index[0]
    distances = similarity_cosine[movie_index]
    recommended_movies = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in recommended_movies:
        print(movies_dataFrame.iloc[i[0]].title)
        

#AUTOENCODER recommendation function
def recommend_autoencoder(movie):
    if movie not in movies_dataFrame['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    
    movie_index = movies_dataFrame[movies_dataFrame['title'] == movie].index[0]
    distances = similarity_cos[movie_index]
    recommended_movies = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in recommended_movies:
        print(movies_dataFrame.iloc[i[0]].title)

# Evaluate all models
bow_precision, bow_recall = evaluate_model(recommend, test_movies, true_relevant)
bert_precision, bert_recall = evaluate_model(recommendCosine, test_movies, true_relevant)
autoencoder_precision, autoencoder_recall = evaluate_model(recommend_autoencoder, test_movies, true_relevant)

# Data for plotting
models = ['BoW', 'BERT', 'Autoencoder']
precisions = [bow_precision, bert_precision, autoencoder_precision]
recalls = [bow_recall, bert_recall, autoencoder_recall]

# Plot Precision and Recall
x = range(len(models))

plt.figure(figsize=(10, 6))
plt.bar(x, precisions, width=0.4, label='Precision', align='center', alpha=0.8)
plt.bar(x, recalls, width=0.4, label='Recall', align='edge', alpha=0.8)

plt.xticks(x, models)
plt.xlabel("Models")
plt.ylabel("Scores")
plt.title("Precision and Recall Comparison")
plt.legend()
plt.show()


# %%
