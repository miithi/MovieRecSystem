#%%
import numpy as np
import pandas as pd

import ast #for objects as string to form a list - abstract syntax tree
import nltk # natural language toolkit 
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity #for the cosine distance between vectors / movies

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on = 'title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords' , 'cast', 'crew']]

#preprocessing
movies.isnull().sum()
movies.dropna(inplace=True)

#dulplicate checker
#movies.duplicated.sum() - no duplicates found so no need for this in this data set

#checking for info before merging
#movies.iloc[0].genres

def convert(obj):
    list = []
    for i in ast.literal_eval(obj):
        list.append(i['name'])
    return list

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_three(obj):
    list = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            list.append(i['name'])
            counter += 1
        else:
            break
    return list

movies['cast'] = movies['cast'].apply(convert_three)

def fetch_director(obj):
    list = []
    for i in ast.literal_eval(obj):
        if i ['job'] == 'Director':
            list.append(i['name'])
            break
    return list

movies['crew'] = movies['crew'].apply(fetch_director)

#just because I am going to concatenate the lists later
# I need a list for overview as well
movies['overview'] = movies['overview'].apply(lambda x:x.split())
#movies.overview[0]


#removing spaces so that the recommender system does not get confused between two entities
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

#now concatenating all the needed columns to produce appropriate tags for the rec system
movies['tags'] = movies['overview'] + movies['genres'] + movies['cast'] + movies['crew'] + movies['keywords']
#movies.head(1)

#creating a brand new data frame which is easier to use
movies_dataFrame = movies[['movie_id', 'title', 'tags']]

#concatenating the list items into one list now 
movies_dataFrame['tags'] = movies_dataFrame['tags'].apply(lambda x:" ".join(x))
#movies_dataFrame['tags'][0]

#suggested update - make everything lowercase
movies_dataFrame['tags'] = movies_dataFrame['tags'].apply(lambda x: x.lower())



#Text Vectorization
#max numbers of top words = 5000
ps = PorterStemmer()

def stem(text):
    list = []
    for i in text.split():
        list.append(ps.stem(i))

    return " ".join(list)

movies_dataFrame['tags'] = movies_dataFrame['tags'].apply(stem)
movies_dataFrame['tags'][0]

#removing english stop words like is, a, are 
cv = CountVectorizer(max_features=5000, stop_words='english') 
vectors = cv.fit_transform(movies_dataFrame['tags']).toarray()
#cv.get_feature_names_out

#similarity score between vectors / movies
similarity = cosine_similarity(vectors)
#similarity[1]

#Recommender function
def recommend(movie):
    movie_index = movies_dataFrame[movies_dataFrame['title'] == movie].index[0]
    distances = similarity[movie_index]
    recommended_movies = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in recommended_movies:
        print(movies_dataFrame.iloc[i[0]].title)

#testing
test_movies = ["Inside Out", "Avatar", "A Christmas Carol", "Inception", "The Dark Knight"]
for movie in test_movies:
    print("\n" + "Recommendations based on: " + movie + "\n")
    recommend(movie)
    print("\n" + "=" * 50 + "\n")
# %%
