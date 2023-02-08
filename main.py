import numpy
import numpy as np
import pandas as pd
import array as arr
import time

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
__category__ = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
                "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War",
                "Western"]
USER_N = 50
FILM_N = 9719
start_time = time.time()


def generate_like_matrix(urm, movie, a=2.5, dislike=False):
    user_like_dic = {}
    LikeSet = {}
    for userId in range(USER_N):
        if dislike:
            film_liked = urm.columns[urm.iloc[userId] < a ].values
        else:
            film_liked = urm.columns[urm.iloc[userId] >= a+1].values
        for movieId in film_liked:
            genre_per_film = set(movie.loc[movie['movieId'] == movieId]['genres'].values[0].split("|"))
            user_like_dic[movieId] = genre_per_film
        LikeSet[userId + 1] = user_like_dic
        if userId==0 and dislike==True:
            print(LikeSet[1])
        user_like_dic = {}
    return LikeSet



def SemSimI(film_au, film_u):
    # F11: Categorie che piacciono ad entrambi gli utenti
    f11 = len(film_au & film_u)
    # F10: Categorie che piacciono solo ad AU
    f10 = len(film_au - film_u)
    # F01: Categorie che piacciono solo ad U
    f01 = len(film_u - film_au)
    return f11 / (f11 + f10 + f01)


# Funzione di SemSimPlus e SemSimMinus (per convenzione i nomi fanno rimento solo al plus)
def sem_sim(like_set_au, like_set_u):
    # Numeratore dell'ecquazione di SemSim
    sum_of_sem_sim_i = 0
    for i_au in like_set_au:
        # Fisso l'active user (au)
        for i_u in like_set_u:
            sum_of_sem_sim_i += SemSimI(like_set_au[i_au], like_set_u[i_u])
    # Denominatore dell'ecquazione di SemSim
    len_au = len(like_set_au)
    len_u = len(like_set_u)
    if len_au == 0:
        len_au = 1
    if len_u == 0:
        len_u = 1
    card_like_set = len_au * len_u
    return sum_of_sem_sim_i / card_like_set

def common_ratings(au, u):
    return urm.columns[urm.iloc[au] != 0 and urm.iloc[u] != 0]
def pearson(au, u, CommonRatings ):
    mean_rating_au = np.mean(urm[au-1,:])
    mean_rating_u = np.mean(urm[u-1,:])
    numeratore,denominatore_au,denominatore_u=0
    for i in CommonRatings:
        numeratore += ((urm[au-1,i-1] - mean_rating_au)* (urm[u-1,i-1] - mean_rating_u))
        denominatore_au += (urm[au-1,i-1] - mean_rating_au)**2
        denominatore_u += (urm[u-1,i-1] - mean_rating_u)**2
    denominatore_au=numpy.sqrt(denominatore_au)
    denominatore_u=numpy.sqrt(denominatore_u)

    return numeratore/denominatore_au*denominatore_u


ratings = pd.read_csv('ml-latest-small/ratings.csv')
movie = pd.read_csv('ml-latest-small/movies.csv')

review_film = ratings.merge(movie, on='movieId', how='left')

print("==========================FILM WITH REVIEW========================")
print(review_film.head(10))

urm = review_film.pivot_table(index='userId', columns='movieId', values='rating')
print("==========================User Rating Matrix========================")
print(urm)

LikeSet = generate_like_matrix(urm, movie)
DisLikeSet = generate_like_matrix(urm, movie, dislike=True)


SemSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
PearsonSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
for i in range(1, USER_N + 1):
    for j in range(1, USER_N + 1):
        SemSimPlus = sem_sim(LikeSet[i], LikeSet[j])
        SemSimMinus = sem_sim(DisLikeSet[i], DisLikeSet[j])
        SemSim = (SemSimPlus + SemSimMinus) / 2
        # Per ogni coppia di utenti (i,j), calcolo il suo valore di similarità con SemSim
        SemSimMatrix.iloc[i - 1][j] = SemSim
        #CommonRatings = common_ratings(i,j)
        #SimPearson = pearson(i,j,CommonRatings)
#print(SemSimMatrix)
#print("--- %s seconds ---" % (time.time() - start_time))

