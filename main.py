import math
import random
import pandas as pd
import nltk
import requests
from sklearn.metrics.pairwise import cosine_similarity
from nltk.chunk import tree2conlltags
from bs4 import BeautifulSoup
from joblib import load

from Initialise import initialise

state = ["Base", "base"]

states = {
    "Base": ["base", "small_talk"],
    "Recommend Movie": ["start", "get_name", "search", "search_end", "recommend_end", "add_recommended", "ask_question",
                        "end"],
    "Watch List Add": ["start", "get_name", "search", "search_end"],
    "Watch List Remove": ["start", "get_name", "search"],
    "Display Watch List": ["start", "get_name", "search", "question"],
    "Change Name": "start"
}

small_talk_generic = {

    "Neutral": ["That’s good", "That’s normal", "Alright, I see", "Okay, got it", "That’s fair",
                "That makes sense", "Okay, sounds good", "Alright, I hear you", "Okay, understood"],
    "Positive": ["That’s great", "That’s awesome", "Wonderful", "Amazing", "Glad to hear that",
                 "That’s fantastic", "Good to hear", "That’s lovely", "So happy to hear that"],
    "Negative": ["That’s unfortunate", "I’m sorry to hear that", "Oh no", "That’s not good", "That’s rough",
                 "I hope it gets better", "That’s upsetting", "Sorry about that", "That’s tough"]

}

user_item_matrix = pd.DataFrame()
watch_list = []
searched_movies = []
recommended_movies = []
question = ""
current_movie = ""
name = ""


# Used to initialise the user item matrix
# initialise() initialises the count vectorizers and transformers and gets all the movie data needed
# initialise() only needs to be run the first time
def startup():

    global user_item_matrix
    initialise()
    user_item_matrix = load("user_item_matrix.joblib")


# Returns the genres of the movie given
def get_genre(movie):

    url = f"https://letterboxd.com/film/{movie}/genres/"
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    string = str(soup)
    start = string.split('"genre":')

    if len(start) < 1:
        return ""

    end = start[1].find("]")
    string = start[1][:end+1]
    list = eval(string)

    return list


# Returns the directors of the movie given
def get_director(movie):
    url = f"https://letterboxd.com/film/{movie}/crew/"
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    string = str(soup)
    names = string.split('<span class="prettify">')

    if len(names) < 2:
        return ""

    directors = []

    for name in names[1:]:
        end = name.find("<")
        directors.append(name[:end])

    return directors


# Returns the runtime of the movie given
def get_runtime(movie):
    url = f"https://letterboxd.com/film/{movie}/crew/"
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    string = str(soup)
    start = string.find("runTime: ")

    if start == -1:
        return ""

    string = string[start+9:]
    end = string.find(" ")
    string = string[:end]

    return string


# Returns the cast of the movie given
def get_cast(movie):
    url = f"https://letterboxd.com/film/{movie}/"
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    string = str(soup)
    string = string.split('<div class="cast-list')

    if len(string) < 2:
        return ""

    string = string[1]
    titles = string.split('">')

    names = []
    max = 10
    for i in range(2, max + 2):
        name = titles[i]
        end = name.find('<')
        names.append(name[:end])
        if "\n" in name:
            break

    return names


# Finds the similarity between the movie given and all other movies
# Returns the results in order of greatest similarity (lowest distance)
def find_closest(movie_search):

    results = []

    for movie in user_item_matrix:
        if movie != movie_search:
            x = user_item_matrix[movie_search]
            y = user_item_matrix[movie]
            distance = math.dist(x, y)
            item = [distance, movie]
            results.append(item)
    sorted_results = sorted(results, key=lambda x: x[0])
    return sorted_results


# Used to sort the results of similarity matching
def sort_results(results):
    sorted_results = []
    pos = 1

    for i in results:
        item = [i[0], pos]
        sorted_results.append(item)
        pos += 1

    return sorted(sorted_results, key=lambda x: x[0], reverse=True)


# Detects the sentiment of a piece of text
def sentiment_analysis(text):
    subj_data = pd.read_csv("corpus/subj_analysis.txt")
    tokens = nltk.word_tokenize(text)
    positive = 0
    negative = 0

    for word in tokens:
        word = word.lower()

        if word in subj_data["Word"].values:
            if subj_data[subj_data["Word"] == word]["Priorpolarity"].values[0] == "positive":
                positive += 1
            elif subj_data[subj_data["Word"] == word]["Priorpolarity"].values[0] == "negative":
                negative += 1

    sum = positive - negative

    if sum == 0:
        return "Neutral"
    elif sum > 0:
        return "Positive"
    else:
        return "Negative"


# Uses named entity recognition to get a name from the text
def extract_name(text):
    text = text.title()
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    entities = nltk.ne_chunk(tagged_tokens)
    tags = tree2conlltags(entities)

    name = ""

    for i in tags:

        if i[2] == "B-PERSON":
            results, input_type = matching(i[0], "name")
            if results[0][0] < 0.2:
                name = i[0]

        elif i[2] == "I-PERSON":
            results, input_type = matching(i[0], "name")
            if results[0][0] < 0.2:
                if name == "":
                    name = i[0]
                else:
                    name += " " + i[0]
    return name


# returns the similarity between a text and all inputs in a file
def matching(text, file):

    count_vect, tf_transformer, X_train_tf, input_type = load("count_vectorizers/" + file + ".joblib")
    input_count = count_vect.transform([text])
    input_count_id = tf_transformer.transform(input_count)

    results = cosine_similarity(X_train_tf, input_count_id)
    return sort_results(results), input_type


# Prints what the chatbot can do
def print_discoverability():
    print(name + ", these are the things I can do!:"
                 "\n\t* Engage in small talk"
                 "\n\t* Answer what your name is and change it"
                 "\n\t* Answer questions"
                 "\n\t* Recommend movies"
                 "\n\t* Add movies to a watch list"
                 "\n\t* Delete movies from watch list"
                 "\n\t* Display watch list"
                 "\n\t* Get information on movies being searched and movies in watch list"
                 "\n\t\t* Cast"
                 "\n\t\t* Runtime"
                 "\n\t\t* Genres"
                 "\n\t\t* Director")


# Checks if the text matches to a variation of exit and goes to base state if it does
def check_exit(text):

    results, input_type = matching(text, "exit")
    if results[0][0] > 0.8:

        print("Leaving the transaction, if you would like to leave the chat then repeat the input"
              "\nWhat else could I do for you?")
        state[0] = "Base"
        state[1] = states["Base"][0]


# takes a movie as the input and uses the global variable question to print an answer to the query
def answer_movie_question(movie):

    movie_name = get_movie_name(movie)

    if question == "Genre":

        genres = get_genre(movie)
        if genres == "":
            print("Sorry there was no genres found for this movie")

        print("The genre(s) of " + movie_name + " are:")
        for genre in genres:
            print("\t*" + genre)

    elif question == "Director":

        directors = get_director(movie)
        if directors == "":
            print("Sorry there was no director found for this movie")
            return

        print("The director(s) of " + movie_name + " are:")
        for director in directors:
            print("\t*" + director)

    elif question == "Runtime":

        runtime = get_runtime(movie)
        if runtime == "":
            print("Sorry there was no runtime found for this movie")
            return

        print("The runtime of " + movie_name + " is " + runtime + " minutes")

    elif question == "Cast":

        cast = get_cast(movie)
        if cast == "":
            print("Sorry there was no cast found for this movie")
            return

        print("The cast of " + movie_name + " is:")

        for member in cast:
            print("\t*"+member)


# Handles a user asking a question depending on the state the chatbot is in
def direct_movie_question(movie_question):

    global question
    question = movie_question

    if state[1] == "add_recommended" or state[1] == "recommend_end":
        print("Which of the recommended movies would you like to know about?")
        state[1] = states["Recommend Movie"][6]
    elif state[1] == "search_end":
        answer_movie_question(searched_movies[0])
        print("Is this the movie you were looking for\n\n\t" + get_movie_name(searched_movies[0]) + "\n")
    else:
        print("Please search for the movie first")


# Prints the watch list
def display_list():

    print(name + ", this is your watch list:\n")
    index = 1
    for movie in watch_list:
        print("\t" + str(index) + "." + get_movie_name(movie))
        index += 1
    print("")


# Display Watch List Transaction
# Handles printing the watch list and then answering queries on it
def display_watch_list(text):

    global state
    global current_movie
    global question

    check_exit(text)

    if state[1] == "start":
        display_list()
        print("Would you like to ask any questions about any of the movies?")
        state[1] = states["Display Watch List"][1]

    elif state[1] == "get_name":

        response = confirmation(text)

        if response == "Yes":

            print("What movie would you like to ask about, please specify or the number on the list")
            state[1] = states["Display Watch List"][2]

        elif response == "No":

            print("What else could I do for you?")
            state[0] = "Base"
            state[1] = states["Base"][0]

    elif state[1] == "search":

        current_movie = specify_results(text)
        if current_movie == "":
            print("Sorry I could not find that movie, please try another input")
            return

        print("What would you like to know?")
        state[1] = states["Display Watch List"][3]

    elif state[1] == "question":

        movie_questions, input_type = matching(text, "movie_questions")

        if movie_questions[0][0] > 0.7:

            question = input_type[movie_questions[0][1] - 1]
            answer_movie_question(current_movie)
            print("Would you like to ask another question?")
            state[1] = states["Display Watch List"][1]

        else:
            print("Sorry i could not recognise that question, please try another input")


# Remove Movie from Watch List Transaction
# Handles input for removing movies from watch list
def remove_movie(text):
    global state
    global current_movie
    global question

    check_exit(text)

    movie_questions, input_type = matching(text, "movie_questions")

    if movie_questions[0][0] > 0.7:
        if state[1] == "get_name":
            print("Please search a movie from the watch list first")
        else:

            question = input_type[movie_questions[0][1] - 1]
            answer_movie_question(current_movie)
            print("Would you like to remove this movie?")

    elif state[1] == "start":
        display_list()
        print("What movie would you like to remove?")
        state[1] = states["Watch List Remove"][1]

    elif state[1] == "get_name":

        current_movie = specify_results(text)
        if current_movie == "":
            print("Sorry I could not find that movie, please try another input")
            return

        print("Is " + get_movie_name(current_movie) + ", the movie you would like to remove?")
        state[1] = states["Watch List Remove"][2]

    elif state[1] == "search":

        response = confirmation(text)

        if response == "Yes":
            watch_list.remove(current_movie)
            print("Movie was removed from your watch list"
                  "\nWhat else could I do for you?")
            state[0] = "Base"
            state[1] = states["Base"][0]

        elif response == "No":
            print("Movie was not removed from your watch list"
                  "\nWhat else could I do for you?")
            state[0] = "Base"
            state[1] = states["Base"][0]

        else:
            print("Could you try a different input please")
            state[1] = states["Watch List Remove"][1]


# Add Movie to Watch List Transaction
# Handles input for adding movies to watch list
def add_movie(text):

    global state

    check_exit(text)

    movie_questions, input_type = matching(text, "movie_questions")
    if movie_questions[0][0] > 0.8:
        direct_movie_question(input_type[movie_questions[0][1] - 1])

    elif state[1] == "start":

        print("What movie would you like to add")
        state[1] = states["Watch List Add"][1]

    elif state[1] == "get_name":

        results, movie_name = matching(text, "movies")
        get_top_results(results, movie_name)
        search_movie_results()

    elif state[1] == "search":

        specify_results(text)
        search_movie_results()

    elif state[1] == "search_end":

        response = confirmation(text)
        if response == "Yes":
            if searched_movies[0] in watch_list:
                print("Movie is already in your watch list")

            else:
                watch_list.append(searched_movies[0])
                print("Movie was added to your watch list")

        elif response == "No":
            print("Movie was not added to your watch list")

        else:
            return

        print("What else could I do for you?")
        state[0] = "Base"
        state[1] = states["Base"][0]


# Recommend Movie Transaction
# Handles input for Recommending Movies
# Adds recommended movies to watch list
def recommend_movie(text):

    global state
    global question

    check_exit(text)

    movie_questions, input_type = matching(text, "movie_questions")
    if movie_questions[0][0] > 0.8:
        direct_movie_question(input_type[movie_questions[0][1] - 1])

    elif state[1] == "start":
        print("Give me the name of a movie you like")
        state[1] = states["Recommend Movie"][1]

    elif state[1] == "get_name":

        results, movie_name = matching(text, "movies")
        get_top_results(results, movie_name)
        search_movie_results()

    elif state[1] == "search":

        specify_results(text)
        search_movie_results()

    elif state[1] == "search_end":

        response = confirmation(text)
        if response == "Yes":
            state[1] = states["Recommend Movie"][4]
            print("Generating recommendations...")
            get_recommended_movie()
        elif response == "No":
            print("Sorry I dont think I have the movie you were looking for"
                  "\nWhat else could I do for you? ")
            state[0] = "Base"
            state[1] = states["Base"][0]

    elif state[1] == "recommend_end":

        response = confirmation(text)
        if response == "Yes":

            print("What movie would you like to add to your watch list?")
            state[1] = states["Recommend Movie"][5]

        elif response == "No":
            print("What else could I do for you? ")
            state[0] = "Base"
            state[1] = states["Base"][0]

    elif state[1] == "add_recommended":

        add_recommended(text)

    elif state[1] == "ask_question":

        movie = specify_results(text)
        if movie == "":
            print("Sorry could not find that movie in the recommended list"
                  "\nPlease try another input")
            return

        answer_movie_question(movie)
        print("Would you like to add a movie to your watch list?")
        state[1] = states["Recommend Movie"][4]


# Change Name Transaction
def change_name(text):

    global name
    check_exit(text)

    new_name = extract_name(text)
    if new_name == "":
        print("Sorry I could not find a name in that response"
              "\nCould you try another input please")
    else:
        print(new_name + " is your new name"
                         "\nWhat else could I do for you?")
        name = new_name
        state[0] = "Base"
        state[1] = states["Base"][0]


# Small Talk
# If low similarity it will print a generic reponse
# Otherwise response is the input pair and based on sentiment if it has different sentiments
def small_talk(text):

    results, sentiments = matching(text, "small_talk")

    if results[0][0] < 0.1:
        sentiment = sentiment_analysis(text)
        answer = small_talk_generic[sentiment][random.randrange(len(small_talk_generic[sentiment]))]
        print(answer)
        state[1] = states["Base"][0]
        return

    if results[0][0] == results[1][0] and results[0][0] == results[2][0]:

        sentiment = sentiment_analysis(text)

        for i in range(3):
            if sentiments[results[i][1] - 1] == sentiment:
                answer = results[i][1] - 1

    else:
        answer = results[0][1] - 1

    small_talk_data = pd.read_csv("corpus/small_talk.txt")
    print(small_talk_data["Answer"][answer])


# Handles adding a recommended movie to watch list
def add_recommended(text):
    # The movie asked for is searched from the list of recommended movies
    movie = specify_results(text)

    # If the movie is not found another input is asked for
    if movie == "":
        print("Sorry could not find that movie in the recommended list"
              "\nPlease try another input")
        return

    # Checks the movie found is not already in the watch list
    if movie in watch_list:
        print("This movie is already in your watch list"
              "\nPlease try another input")
        return

    # Valid movie selected, so it is added to the watch list
    watch_list.append(movie)
    print("This movie has been added to your watch list")

    # Checks that all the movies in the recommended list have not already been added to watch list
    found = 0
    for list_movie in recommended_movies:
        if list_movie in watch_list:
            found += 1

    # Puts the user back in the original state if all movies have been added
    if found == len(recommended_movies):
        print("All movies in the recommended list have already been added to your watch list"
              "\nWhat else could I do for you? ")
        state[0] = "Base"
        state[1] = states["Base"][0]
        return

    print("Would you like to add another movie to your watch list?")
    state[1] = states["Recommend Movie"][4]


# Adds the movies with a high similarity to a global searched list
def get_top_results(results, movie_name):

    global searched_movies
    searched_movies = []
    for similarity in results:

        if similarity[0] < 0.6:
            break

        searched_movies.append(movie_name[similarity[1] - 1])


# Handles different search result sizes to narrow down results
# If no results then goes back to base state
# If one result it uses confirmation
# If multiple results it prints them and asks user to specify
def search_movie_results():
    if len(searched_movies) == 0:
        print("Sorry i could not find that movie, could you try a different input")
        state[1] = states[state[0]][1]
        return

    elif len(searched_movies) == 1:
        print("Is this the movie you were looking for\n\n\t" + get_movie_name(searched_movies[0]) +"\n")
        state[1] = states[state[0]][3]

    else:
        print("Here are the results found:\n")
        i = 1
        for movie in searched_movies:
            name = get_movie_name(movie)
            print("\t" + str(i) + ". " + name)
            i += 1

        print("\nCould you please specify the movie or the number on the list")
        state[1] = states[state[0]][2]


# Returns movie name based on unique name that is stored in lists
def get_movie_name(ID):
    movie_names_data = pd.read_csv("corpus/movies.txt")
    return movie_names_data[movie_names_data["movieID"] == ID]["movie_name"].values[0]


# Narrows results down to one by either returning result with highest similarity
# Or if none are similar it is based on the input being the number in the results
# If not nothing is found
def specify_results(text):

    global searched_movies
    global recommended_movies

    results, movie_name = matching(text, "movies")
    movie_found = movie_name[results[0][1] - 1]

    found = False
    index = 0

    if state[0] == "Display Watch List" or state[0] == "Watch List Remove":
        movies = watch_list

    elif state[1] == "search":
        movies = searched_movies

    elif state[1] == "add_recommended" or "recommend_end":
        movies = recommended_movies

    for movie in movies:

        if movie_found == movie:
            found = True
            break
        index += 1

    if found:
        if state[1] == "search":
            searched_movies = [movies[index]]

        return movies[index]

    if text.isdigit():
        index = int(text)
        if index <= len(movies):

            if state[1] == "search":
                searched_movies = [movies[index - 1]]

            return movies[index - 1]

    searched_movies = []
    return ""


# Checks if text is matching to some variation of either yes or no
# If not the input is not recognised
def confirmation(text):

    results, input_type = matching(text, "confirmation")

    if results[0][0] < 0.5:
        print("That was not a recognised response"
              "\nCould you try another input")
        return "None"

    if input_type[results[0][1] - 1] == "Yes":
        return "Yes"
    else:
        return "No"


# Gets a list of all movies in order of similarity to searched movie
# Then displays the first five not already in the users' watch list
def get_recommended_movie():

    results = find_closest(searched_movies[0])

    print(name + ", these are my top five movie recommendations:")

    added = 0
    for i in range(0, 5 + added):
        movie = results[i][1]
        if movie in watch_list:
            added += 1
        else:
            movie_name = get_movie_name(movie)
            print("\n\t" + str(i - added + 1) + ". " + movie_name)

        recommended_movies.append(movie)

    print("\nWould you like to add any of these to your watch list")


# Returns the intent with the highest similarity to the text
def intent_matching(text):

    st_results, st_input_type = matching(text, "small_talk")
    QA_results, QA_input_type = matching(text, "QA")
    disc_results, disc_input_type = matching(text, "discoverability")
    name_results, name_input_type = matching(text, "name")
    list_results, list_input_type = matching(text, "list_matching")

    high = max_list([st_results[0][0], QA_results[0][0], disc_results[0][0], name_results[0][0], list_results[0][0]])

    if high == st_results[0][0]:
        return st_results, st_input_type, 1
    elif high == disc_results[0][0]:
        return disc_results, disc_input_type, 2
    elif high == name_results[0][0]:
        return name_results, name_input_type, 3
    elif high == QA_results[0][0]:
        return QA_results, QA_input_type, 4
    elif high == list_results[0][0]:
        return list_results, list_input_type, 5


# Return max if list is not empty otherwise returns 0
def max_list(list):
    if len(list) > 0:
        return max(list)
    else:
        return 0


# Main Loop
# Handles all input based on the state and intent
def main():

    global state
    global name

    response = input("Hello what is your name ")
    name = extract_name(response)
    print("Hello " + name + " what can I do for you?")

    while True:

        user_input = input()

        if state[0] == "Recommend Movie":
            recommend_movie(user_input)
            continue

        elif state[0] == "Watch List Add":
            add_movie(user_input)
            continue

        elif state[0] == "Display Watch List":
            display_watch_list(user_input)
            continue

        elif state[0] == "Watch List Remove":
            remove_movie(user_input)
            continue

        elif state[0] == "Change Name":
            change_name(user_input)
            continue

        exit_results, exit_input_type = matching(user_input, "exit")
        if exit_results[0][0] > 0.8:
            print("Closing the chat"
                  "\nThank you for using the chatbot")
            break

        results, input_type, mode = intent_matching(user_input)
        ID = results[0][1] - 1

        if state[1] == "base":
            if results[0][0] < 0.3:
                print("Sorry I could not understand that input")
                continue

            elif mode == 1:

                state[1] = states["Base"][1]
                small_talk(user_input)
                continue

        elif state[1] == "small_talk":
            if not mode == 1 and results[0][0] > 0.6:
                state[1] = states["Base"][0]
            else:
                small_talk(user_input)
                if not state[1] == "small_talk":
                    print("What else can I do for you?")
                continue

        if mode == 2:

            print_discoverability()

        elif mode == 3:

            if input_type[ID] == "AskName":
                print("Your name is " + name)
            else:
                new_name = extract_name(user_input)
                if new_name == "":
                    state[0] = "Change Name"
                    state[1] = states["Change Name"][0]

                    print("What would you like your new name to be?")
                    continue
                else:
                    print(new_name + " is your new name")
                    name = new_name

        elif mode == 4:

            print(input_type[ID])

        elif mode == 5:

            intent = input_type[ID]

            if intent == "AddWatchList":

                state = ["Watch List Add", states["Watch List Add"][0]]

                add_movie("")

            elif intent == "RemoveWatchList":

                if len(watch_list) == 0:
                    print("Watch list is empty, please add to it first"
                          "\nWhat else can I do for you?")
                    continue

                state = ["Watch List Remove", states["Watch List Remove"][0]]
                remove_movie(user_input)

            elif intent == "DisplayWatchList":

                if len(watch_list) == 0:
                    print("Watch list is empty, please add to it first"
                          "\nWhat else can I do for you?")
                    continue

                state = ["Display Watch List", states["Display Watch List"][0]]
                display_watch_list(user_input)

            elif intent == "RecommendMovie":

                state = ["Recommend Movie", states["Recommend Movie"][0]]
                recommend_movie("")

        if not mode == 5 and not state[1] == "small_talk":
            print("What else can I do for you?")


startup()
main()
