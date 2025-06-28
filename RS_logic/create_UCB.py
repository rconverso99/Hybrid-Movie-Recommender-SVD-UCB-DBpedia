import numpy as np
import math
import pandas as pd
from scipy.sparse import csr_matrix


def create_urm_from_data(data, user_ids, movie_ids):
    """
    Crea la matrice URM a partire dai dati e dagli ID di utenti e film.
    Utilizza una matrice sparsa per efficienza di memoria.

    Args:
        - data: una lista di tuple (id_utente, id_film, valutazione).
        - user_ids: lista degli ID unici degli utenti.
        - movie_ids: lista degli ID unici dei film.

    Returns:
        - urm: matrice URM sparsa (users x movies).
        - user_map: mappa degli ID utenti a indici numerici.
        - movie_map: mappa degli ID film a indici numerici.
    """
    # Creare mappe da ID a indici
    user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    # Preparare liste per costruire la matrice sparsa
    row_indices = []
    col_indices = []
    ratings = []

    # Popolare le liste con i dati
    for user_id, movie_id, rating in data:
        if user_id in user_map and movie_id in movie_map:
            user_idx = user_map[user_id]
            movie_idx = movie_map[movie_id]
            row_indices.append(user_idx)
            col_indices.append(movie_idx)
            ratings.append(rating)

    # Creare matrice sparsa
    urm = csr_matrix((ratings, (row_indices, col_indices)),
                     shape=(len(user_ids), len(movie_ids)))

    return urm, user_map, movie_map


def calculate_UCB(urm, movie_map, l=2):
    """
    Calcola l'indice UCB per ogni film nella matrice URM sparsa.

    Args:
        - urm: matrice URM sparsa (users x movies) dove ogni valore è una valutazione o 0 se non valutato.
        - movie_map: mappa degli ID film a indici numerici.
        - l: fattore di esplorazione (default 2).

    Returns:
        - ucb_pairs: un array di coppie (movieId, UCB).
    """
    num_users, num_movies = urm.shape

    # Invertiamo movie_map per ottenere indice -> movieId
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_map.items()}

    # Calcoliamo le somme e il numero di valutazioni per ogni film
    sum_ratings = np.array(urm.sum(axis=0)).flatten()
    count_ratings = np.array((urm != 0).sum(axis=0)).flatten()

    # Calcoliamo la media delle valutazioni per ogni film
    avg_ratings = np.zeros(num_movies)
    nonzero_mask = count_ratings > 0
    avg_ratings[nonzero_mask] = sum_ratings[nonzero_mask] / count_ratings[nonzero_mask]

    # Array per memorizzare le coppie (movieId, UCB)
    ucb_pairs = []

    # Calcoliamo l'UCB per ogni film
    for movie_idx in range(num_movies):
        movie_id = idx_to_movie[movie_idx]
        num_selections = count_ratings[movie_idx]

        if num_selections > 0:
            ucb = avg_ratings[movie_idx] + np.sqrt(l * np.log(num_users) / num_selections)
        else:
            ucb = float('inf')

        # Aggiungiamo la coppia (movieId, UCB)
        ucb_pairs.append((movie_id, ucb))

    return np.array(ucb_pairs, dtype=object)

# Carica i dati
file = pd.read_csv("../CSV_files/ratings.csv", index_col=False)

# Ottieni liste di ID univoci
unique_user_ids = sorted(file["userId"].unique())
unique_movie_ids = sorted(file["movieId"].unique())

print(f"Numero di utenti unici: {len(unique_user_ids)}")
print(f"Numero di film unici: {len(unique_movie_ids)}")

# Crea una lista di tuple dai dati
data_example = list(file[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))

# Creare la matrice URM
urm_example, user_map, movie_map = create_urm_from_data(data_example, unique_user_ids, unique_movie_ids)
print(f"Dimensioni della matrice URM: {urm_example.shape}")
print(f"Memoria utilizzata dalla matrice (MB): {urm_example.data.nbytes / (1024 * 1024):.2f}")

# Calcolare l'UCB per ogni film
ucb_result = calculate_UCB(urm_example, movie_map, l=2)

# Stampa i primi 10 valori UCB
print("Primi 10 UCB per i film:")
#for i in range(10):
    #print(f"{ucb_result[i]}")

# Salva il DataFrame come CSV
#ucb_df = pd.DataFrame(ucb_result, columns=["movieId", "ucbId"])
#csv_filename = "../CSV_files/movies_ucb.csv"
#ucb_df.to_csv(csv_filename, index=False)

ucb = pd.read_csv("../CSV_files/movies_ucb.csv", index_col=False)
movies = pd.read_csv("../CSV_files/movies.csv", index_col=False)
merged = movies.merge(ucb, on="movieId", how="left")
merged = merged[['movieId', 'ucbId', 'genres']]
print(merged.head(10))




