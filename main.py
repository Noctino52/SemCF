import numpy as np
import pandas as pd
import math
import time

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
__category__ = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
                "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War",
                "Western"]
# Variabili globali
USER_N = 50
FILM_N = 9719

def generate_like_matrix(urm, movie, a,dislike=False):
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
def toNL(Matrix,Nnibor):
    u = np.argpartition(Matrix, axis=1, kth=Nnibor)
    v = Matrix.columns.values[u].reshape(u.shape)
    NL = pd.DataFrame(v[:, -Nnibor:]).rename(columns=lambda x: f'Max{x + 1}')
    return NL

def generate_prediction(urm,UNL,SimMatrix):
    PredictionMatrix = pd.DataFrame(columns=range(1, 193610), index=range(1, USER_N + 1))
    for au in range(0, USER_N):
        if (urm.index == au).any():
            mean_rating_au = urm.loc[au + 1].mean()
        else:
            mean_rating_au = 0
        for movie in urm.columns[0:]:
            if not math.isnan(urm.loc[au + 1][movie]):
                # print("UTENTE: "+str(au))
                # print("FILM: "+str(movie))
                PredictionMatrix.iloc[au][movie] = predict(SimMatrix, au, UNL[au], movie, urm, mean_rating_au)
            else:
                PredictionMatrix.iloc[au][movie] = 7000
        # Elimino colonne NaN (i movieId non sono contigui)
    PredictionMatrix = PredictionMatrix.dropna(axis=1, how='all')
    return PredictionMatrix


# Funzione per la predizione e il riempimento della urm (FUNZIONANTE (ritorna voti da 0 a 5) )
def predict(SimMatrix, au, SetAu, movie, urm, meanRateAu):
    numeratore = 0
    denominatore = 0
    for u in SetAu:
        mean_rating_u = urm.loc[u].mean()
        if math.isnan(mean_rating_u):
            mean_rating_u = 0
        # Il numeratore si somma soltanto nei casi in cui il film è stato votato
        if not math.isnan(urm.loc[u][movie]):
            votoU = urm.loc[u][movie]
            numeratore += (SimMatrix.iloc[au][u]) * (votoU - mean_rating_u)
        # Il denominatore si calcola a priori
        denominatore += SimMatrix.iloc[au][u]
    # ritorno = (numeratore / denominatore)+mean_rating_au
    # print("Utente N."+str(au+1)+" , predetto il valore " + str(ritorno) + " per l'item " + str(movie))
    return meanRateAu + (numeratore / denominatore)


def generate_UNL(SNL, PNL,Nnibor):
    listPNL = PNL.values.tolist()
    listSNL = SNL.values.tolist()
    UNL = {}
    for i in range(USER_N):
        # Per ogni utente, prendo la lista dei vicini, la trasformo in un set ed effettuo un'intersezione insiemistica
        UNL[i] = set(listPNL[i]) & set(listSNL[i])
        # Se l'intersezione è vuota
        if len(UNL[i]) == 0:
            # Inserisco nella UNL i  NUM_NIBOR/2 migliori valori presenti nelle PNL e SNL
            for k in range(Nnibor, int((Nnibor / 2) + 1), -1):
                UNL[i].add(listSNL[i][k - 1])
            for l in range(Nnibor, int((Nnibor / 2) + 1), -1):
                UNL[i].add(listPNL[i][l - 1])
    return UNL


def mae(urm,PredictionMatrix):
    num_of_entry_err = 0
    sum_of_entry_error = 0
    for u in range(1, USER_N + 1):
        for movie in urm.columns[0:]:
            r_predicted = PredictionMatrix.loc[u][movie]
            r_real = urm.loc[u][movie]
            if not math.isnan(r_predicted) and not math.isnan(r_real) and r_predicted != 7000:
                entry_error = abs(r_predicted - r_real)
                sum_of_entry_error += entry_error
                num_of_entry_err += 1
    return sum_of_entry_error / num_of_entry_err

#MODEL BUILIDING

# IMPORT FILE CSV
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movieList = pd.read_csv('ml-latest-small/movies.csv', nrows=5001)
colnames = ['movieId', 'title', 'genres']
validation_movie = pd.read_csv('ml-latest-small/movies.csv', skiprows=5001, nrows=2000, names=colnames, header=None)
review_film_validation = ratings.merge(validation_movie, on='movieId', how='inner')
urm_validation = review_film_validation.pivot_table(index='userId', columns='movieId', values='rating')
# Se l'utente non ha film nell'urm_test allora aggiungo l'utente con una riga di NaN
for i in range(1, 611):
    if not (urm_validation.index == i).any():
        urm_validation.loc[i] = float("nan")

review_film = ratings.merge(movieList, on='movieId', how='inner')

# USER RATING MATRIX
urm = review_film.pivot_table(index='userId', columns='movieId', values='rating')
print("GENERATE LE URM")

mina=0
minnn=0
minmae=0
for a in range(4,7):
    a=a/2
    for Nnibor in range(10,41,10):
        start_time = time.time()
        #Generazione LikeSet e DisLikeSet
        LikeSet = generate_like_matrix(urm,movieList,a)
        DisLikeSet = generate_like_matrix(urm, movieList, a, dislike=True)

        print("GENERATI I LIKE SET")
        # Allocazione matrici
        SemSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
        PearsonSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
        JaccardSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))

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

        print("CALCOLATA LA SIMMATRIX")
        # Recupero i NUM_NIBOR elementi massimi di SemSimMatrix e PreSimMatrix
        # SNL (Semantic Neighborhood List)
        # PNL (PreSim Neighborhood List
        SNL = toNL(SemSimMatrix,Nnibor)
        PNL = toNL(PreSimMatrix,Nnibor)

        # UNL (Unified Neighborhood List) : Intersezione di SNL e PNL
        UNL = generate_UNL(SNL, PNL,Nnibor)
        print("CALCOLATA LA UNL")
        # Calcolo dei ratings predetti
        PredictionMatrix= generate_prediction(urm_validation,UNL,SimMatrix)
        print("CALCOLATA LA PREDICTION MATRIX")
        mean_absolute_error= mae(urm_validation, PredictionMatrix)

        #MODEL SELECTION
        if minmae>mean_absolute_error:
            minmae=mean_absolute_error
            mina=a
            minnn=Nnibor

        #ANNOTAZIONE RISULTATI
        f=open("result.txt","a")
        timeF=time.time() - start_time
        del start_time
        f.write("[Time= "+str(timeF)+"] [a= "+str(a)+"] [ NUM_NIBOR= "+str(Nnibor)+"] La MAE E': "+str(mean_absolute_error)+"\n")
        f.close()

        del PredictionMatrix
        del SimMatrix
        del PreSimMatrix



#ADDESTRAMENTO MODELLO MIGLIORE SUL TRAINING COMPLETO
del movieList
movieList = pd.read_csv('ml-latest-small/movies.csv', nrows=7001)
colnames = ['movieId', 'title', 'genres']
test_movie = pd.read_csv('ml-latest-small/movies.csv', skiprows=7001, nrows=2741, names=colnames, header=None)
review_film_test = ratings.merge(test_movie, on='movieId', how='inner')
urm_test = review_film_test.pivot_table(index='userId', columns='movieId', values='rating')
# Se l'utente non ha film nell'urm_test allora aggiungo l'utente con una riga di NaN
for i in range(1, 611):
    if not (urm_test.index == i).any():
        urm_test.loc[i] = float("nan")
del review_film
del urm
review_film = ratings.merge(movieList, on='movieId', how='inner')
# USER RATING MATRIX
urm = review_film.pivot_table(index='userId', columns='movieId', values='rating')

start_time = time.time()
#Generazione LikeSet e DisLikeSet
LikeSet = generate_like_matrix(urm,movieList,mina)
DisLikeSet = generate_like_matrix(urm, movieList, mina, dislike=True)

print("GENERATI I LIKE SET")
# Allocazione matrici
SemSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
PearsonSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))
JaccardSimMatrix = pd.DataFrame(columns=range(1, USER_N + 1), index=range(1, USER_N + 1))

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

print("CALCOLATA LA SIMMATRIX")
# Recupero i NUM_NIBOR elementi massimi di SemSimMatrix e PreSimMatrix
# SNL (Semantic Neighborhood List)
# PNL (PreSim Neighborhood List
SNL = toNL(SemSimMatrix,minnn)
PNL = toNL(PreSimMatrix,minnn)

# UNL (Unified Neighborhood List) : Intersezione di SNL e PNL
UNL = generate_UNL(SNL, PNL,minnn)
print("CALCOLATA LA UNL")
# Calcolo dei ratings predetti
PredictionMatrix= generate_prediction(urm_test,UNL,SimMatrix)
print("CALCOLATA LA PREDICTION MATRIX")

#RISULTATO TEST
mean_absolute_error= mae(urm_test, PredictionMatrix)

#ANNOTAZIONE RISULTATO FINALE
f=open("result.txt","a")
time=time.time() - start_time
f.write("\n\nRISULTATO FINALE: [Time= "+str(time)+"] [a= "+str(mina)+"] [ NUM_NIBOR= "+str(minnn)+"] La MAE E': "+str(mean_absolute_error)+"\n")
f.close()
