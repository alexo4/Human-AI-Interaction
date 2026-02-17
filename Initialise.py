import os
from io import StringIO
from string import punctuation

import pandas as pd
import requests
from bs4 import BeautifulSoup
from joblib import dump, load
import csv

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# Used to intialise all data for the first time
# Get data is commented out due to it taking many hours as it gets a lot of data, hence it is only run once
def initialise():

    #get_data()
    create_vectorizer_transformer()
    movie_data = load('movie_data.joblib')
    df = pd.DataFrame(movie_data)
    user_item_matrix = df.pivot(index=['User'], columns=['Movie'], values='Review').fillna(0)
    dump(user_item_matrix, "user_item_matrix.joblib")


# Gets the data for the user item matrix and dumps the data
# Writes another file with pairs of movie ID and movie name
def get_data():

    users = get_users()
    movie_data = get_all_movies(users)
    dump(movie_data, 'movie_data.joblib')
    write_movies(movie_data)


# Returns a list of all users' ID in the first 50 pages
def get_users():
    users = []

    for i in range(1, 50):
        url = f'https://letterboxd.com/members/popular/page/{i}/'
        temp = get_users_from_page(url)
        users = users + temp

    return users


# Returns the IDs of all users in a given page
def get_users_from_page(url):
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    string = str(soup)
    users = []

    sub = 'avatar -a40" href="/'
    print(string.find(sub))

    i = 0
    while string.find(sub) != -1 and i < 30:
        start = string.find(sub)
        string = string[(start + len(sub))::1]

        end = string.find("/")
        user = string[:end:1]
        users.append(user)
        i += 1

    return users


# Given a list of user IDs it will return a dictionary of all the movies they have rated
def get_all_movies(users):
    user_data = []
    movie_data = []
    review_data = []

    for user in users:
        url = f"https://letterboxd.com/{user}/films/"
        movies, reviews = get_all_movies_from_user(url)
        if movies == "":
            continue
        movie_data = movie_data + movies
        review_data = review_data + reviews
        for i in range(0, len(movies)):
            user_data.append(user)

    data = {
        "User": user_data,
        "Movie": movie_data,
        "Review": review_data,
    }

    return data


# Finds how many pages of reviews a user has gets all movies and reviews from all pages and returns them
def get_all_movies_from_user(url):

    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')

    string = str(soup)
    pages = string.split("/films/page/")
    last_page_str = pages[len(pages) - 1]
    last_page = last_page_str[:last_page_str.find("/"):]

    movies = []
    reviews = []
    if not last_page.isdigit():
        last_page = 1

    if int(last_page) > 70:
        return "", ""

    for i in range(1, int(last_page) + 1):
        temp_url = url + "page/" + str(i)
        movies, reviews = get_all_movies_from_page(temp_url, movies, reviews)

    return movies, reviews


# Returns all movies and review scores for those movies in a page
def get_all_movies_from_page(url, movies, reviews):

    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')

    string = str(soup)
    titles = string.split('data-film-slug="')

    sub = 'micro -darker rated-'

    for title in titles[1::1]:

        end = title.find('"')
        movie = title[:end:1]
        if movie not in movies:
            start = title.find(sub)
            if start != -1:
                title = title[(start + len(sub))::1]
                end = title.find('"')
                score = int(title[:end:1]) / 2
                movies.append(movie)
                reviews.append(score)

    return movies, reviews


#
def write_movies(movie_data):
    movies = list(dict.fromkeys(movie_data["Movie"]))
    movies_names = [i.replace("-", " ") for i in movies]
    rows = zip(movies, movies_names)

    with open("corpus/movies.txt", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["movieID", "movie_name"])
        for row in rows:
            writer.writerow(row)


def create_vectorizer_transformer():
    document_path = "corpus"

    corpus = {}
    for file in os.listdir(document_path):
        filepath = document_path + os.sep + file
        if file == ".DS_Store" or file == "subj_analysis.txt":
            continue
        with open(filepath, encoding='utf8', errors='ignore', mode='r') as document:
            content = document.read()
            document_id = file
            corpus[document_id] = content

    for file in corpus:

        data = pd.read_csv(StringIO(corpus[file]))

        if file == "QA.txt" :
            documents = []
            for i, row in data.iterrows():
                document = ' '.join(row.values[1:])
                documents.append(document)

        else:
            documents = data[data.columns[1]].tolist()

        if file == "QA.txt.":
            lemmatiser = WordNetLemmatizer()
            analyzer = CountVectorizer().build_analyzer()

            def lem_words(doc):
                return (lemmatiser.lemmatize(w).lower() for w in analyzer(doc) if
                        (w not in stopwords.words('english')) and
                        (w not in punctuation))

            count_vect = CountVectorizer(analyzer=lem_words).fit(documents)
        else:
            count_vect = CountVectorizer().fit(documents)

        X_train_counts = count_vect.transform(documents)
        tf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)

        if len(data.columns) > 2:
            input_type = data[data.columns[2]].tolist()
        elif file == "movies.txt":
            input_type = data[data.columns[0]].tolist()
        else:
            input_type = []

        dump((count_vect, tf_transformer, X_train_tf, input_type), "count_vectorizers/"+file[:-4]+".joblib")
















