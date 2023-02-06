import numpy as np
import pandas as pd
import array as arr

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 43999)
pd.set_option('display.width', 2000)
__category__ = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
                "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War",
                "Western"]


def get_top_category_per_user(review_film, userId=0, top=5, flop=False):
    top_category = {}
    for category in __category__:
        top_category["" + category + ""] = 0
    for index, rating in review_film.iterrows():
        categories = rating['genres'].split("|")
        # Conto quanti film di una categoria ha visto
        for c in categories:
            for category in top_category:
                if c == category:
                    top_category["" + category + ""] += 1

    # Se l'utente non ha visto film di quella categoria  elimino il genere dalla top
    for category in __category__:
        if top_category["" + category + ""] == 0:
            top_category.pop("" + category + "")

    # Ordino per valore decrescente, se flop=true crescente, dopodichè prendo solo le prime top categorie
    top_category = sorted(top_category.items(), key=lambda x: x[1], reverse=not flop)[0:top]
    return top_category


def users_trends(review_film, top=5, flop=False):
    user_trends = {}
    last = 0
    for index, rating in review_film.iterrows():
        curr = rating["userId"]
        if last != curr:
            user_trends[curr] = get_top_category_per_user(review_film.query("userId ==" + str(curr)), top=top, flop=flop)
            last = curr

    return user_trends


ratings = pd.read_csv('ml-latest-small/ratings.csv')
movie = pd.read_csv('ml-latest-small/movies.csv')

review_film = ratings.merge(movie, on='movieId', how='left')

print("==========================FILM WITH REVIEW========================")
print(review_film.head(10))

urm = review_film.pivot_table(index='userId', columns='title', values='rating')
print("==========================User Rating Matrix========================")
print(urm.head(10))

# Le categorie=0 vanno contate? in teoria sono film che non piacciono
print("==========================Top User trends========================")
print(users_trends(review_film))
print("==========================Flop User trends========================")
print(users_trends(review_film, flop=True))
