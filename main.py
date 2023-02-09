import numpy as np
import pandas as pd
import math
import time

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000000)
__category__ = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
                "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War",
                "Western"]
# Variabili globali
USER_N = 10
FILM_N = 9719
start_time = time.time()
NUM_NIBOR = 5


def generate_like_matrix(urm, movie, a=2.5, dislike=False):
    user_like_dic = {}
    LikeSet = {}
    for userId in range(USER_N):
        if dislike:
            film_liked = urm.columns[urm.iloc[userId] < a].values
        else:
            film_liked = urm.columns[urm.iloc[userId] >= a].values
        for movieId in film_liked:
            genre_per_film = set(movie.loc[movie['movieId'] == movieId]['genres'].values[0].split("|"))
            user_like_dic[movieId] = genre_per_film
        LikeSet[userId + 1] = user_like_dic
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


# Funzione di SemSimPlus e SemSimMinus (per convenzione i nomi fanno riferimento solo al plus)
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


# Funzione per il calocolo della correlazione di pearson tra due users
def pearson(au, u):
    mean_rating_au = urm.loc[au].mean()
    mean_rating_u = urm.loc[u].mean()
    numeratore = 0
    denominatore_u = 0
    denominatore_au = 0

    for i, j in zip(urm.loc[au], urm.loc[u]):
        if not math.isnan(i) and not math.isnan(j):
            numeratore += (i - mean_rating_au) * (j - mean_rating_u)
            denominatore_au += (i - mean_rating_au) ** 2
            denominatore_u += (j - mean_rating_u) ** 2
    denominatore_au = np.sqrt(denominatore_au)
    denominatore_u = np.sqrt(denominatore_u)
    if ((denominatore_au * denominatore_u) == 0):
        return 0
    else:
        return numeratore / (denominatore_au * denominatore_u)


# Funzione per il calcolo della similarità di Jaccard tra due users
def jaccard(au, u):
    sharedItems = 0
    auItems = 0
    uItems = 0

    for i, j in zip(urm.loc[au], urm.loc[u]):
        if not math.isnan(i) and not math.isnan(j):
            sharedItems = sharedItems + 1
        if not math.isnan(i):
            auItems = auItems + 1
        if not math.isnan(j):
            uItems = uItems + 1

    return sharedItems / (auItems + uItems)


# Funzione che data in ingresso una matrice, restituisce per ogni riga in una matrice i NUM_NIBOR elementi massimi
def toNL(Matrix):
    u = np.argpartition(Matrix, axis=1, kth=NUM_NIBOR)
    v = Matrix.columns.values[u].reshape(u.shape)
    NL = pd.DataFrame(v[:, -NUM_NIBOR:]).rename(columns=lambda x: f'Max{x + 1}')
    return NL


# Funzione per la predizione e il riempimento della urm (FUNZIONANTE (ritorna voti da 0 a 5) )
def predict(SimMatrix, au, SetAu, movie, urm, meanRateAu):
    numeratore = 0
    denominatore = 0
    for u in SetAu:
        mean_rating_u = urm.iloc[u].mean()
        # Il numeratore si somma soltanto nei casi in cui il film è stato votato
        if not math.isnan(urm.loc[u][movie]):
            votoU = urm.loc[u][movie]
            numeratore += (SimMatrix.iloc[au][u]) * (votoU - mean_rating_u)
        # Il denominatore si calcola a priori
        denominatore += SimMatrix.iloc[au][u]
    ritorno = numeratore / denominatore
    print("Utente N."+str(au+1)+" , predetto il valore " + str(ritorno) + " per l'item " + str(movie))
    return meanRateAu + (numeratore / denominatore)


def generate_UNL(SNL, PNL):
    listPNL = PNL.values.tolist()
    listSNL = SNL.values.tolist()
    UNL = {}
    for i in range(USER_N):
        # Per ogni utente, prendo la lista dei vicini, la trasformo in un set ed effettuo un'intersezione insiemistica
        UNL[i] = set(listPNL[i]) & set(listSNL[i])
        # Se l'intersezione è vuota
        if len(UNL[i]) == 0:
            # Inserisco nella UNL i  NUM_NIBOR/2 migliori valori presenti nelle PNL e SNL
            for k in range(NUM_NIBOR, int((NUM_NIBOR / 2) + 1), -1):
                UNL[i].add(listSNL[i][k - 1])
            for l in range(NUM_NIBOR, int((NUM_NIBOR / 2) + 1), -1):
                UNL[i].add(listPNL[i][l - 1])
    return UNL


# IMPORT FILE CSV
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movie = pd.read_csv('ml-latest-small/movies.csv')
review_film = ratings.merge(movie, on='movieId', how='left')

# USER RATING MATRIX
urm = review_film.pivot_table(index='userId', columns='movieId', values='rating')

# Generazione LikeSet e DisLikeSet
LikeSet = generate_like_matrix(urm, movie)
DisLikeSet = generate_like_matrix(urm, movie, dislike=True)

# Allocazione matrici
SemSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
PearsonSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
JaccardSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
PredictionMatrix = pd.DataFrame(columns=range(1, FILM_N + 1), index=range(1, USER_N + 1))

# Calolo SemSim, SimPearson, SimJaccard
for i in range(1, USER_N + 1):
    for j in range(1, USER_N + 1):
        # SemSim
        SemSimPlus = sem_sim(LikeSet[i], LikeSet[j])
        SemSimMinus = sem_sim(DisLikeSet[i], DisLikeSet[j])
        SemSim = (SemSimPlus + SemSimMinus) / 2
        # Per ogni coppia di utenti (i,j), calcolo il suo valore di similarità con le tre funzioni
        SemSimMatrix.iloc[i - 1][j] = SemSim
        # Non ci interessa sapere la somiglianza con lo stesso utente, inseriamo un valore placeholder per riempire la matrice
        if i == j:
            SemSimMatrix.iloc[i - 1][j] = -1
        # Pearson
        SimPearson = pearson(i, j)
        PearsonSimMatrix.iloc[i - 1][j] = SimPearson
        if i == j:
            PearsonSimMatrix.iloc[i - 1][j] = -5
        # Jaccard
        SimJaccard = jaccard(i, j)
        JaccardSimMatrix.iloc[i - 1][j] = SimJaccard
        if i == j:
            JaccardSimMatrix.iloc[i - 1][j] = 1

# Calcolo PreSimMatrix e SimMatrix
PreSimMatrix = JaccardSimMatrix * PearsonSimMatrix
SimMatrix = (PreSimMatrix + SemSimMatrix) / 2

# Recupero i NUM_NIBOR elementi massimi di SemSimMatrix e PreSimMatrix
# SNL (Semantic Neighborhood List)
# PNL (PreSim Neighborhood List
SNL = toNL(SemSimMatrix)
PNL = toNL(PreSimMatrix)

# UNL (Unified Neighborhood List) : Intersezione di SNL e PNL
UNL = generate_UNL(SNL, PNL)

# Calcolo dei ratings predetti
for au in range(0, USER_N):
    mean_rating_au = urm.iloc[au].mean()
    for movie in urm.columns[0:]:
        PredictionMatrix.iloc[au][movie] = predict(SimMatrix, au, UNL[au], movie, urm, mean_rating_au)
    # Elimino colonne NaN (i movieId non sono contigui)
    PredictionMatrix = PredictionMatrix.dropna(axis=1, how='all')
print(PredictionMatrix)

# Recupero i massimi (NON FUNZIONA- gli indici dei massimi in realtà contengono NaN)
Consigliati = toNL(PredictionMatrix)
print(Consigliati)

print("--- %s seconds ---" % (time.time() - start_time))
