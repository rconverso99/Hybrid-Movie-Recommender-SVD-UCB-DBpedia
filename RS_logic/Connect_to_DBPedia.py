import pandas as pd
import re
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from bs4 import BeautifulSoup
import Calculate_user_genre as cug
import Calculate_user_best_actor as cuba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random as rd

global estimated
estimated = pd.read_csv("../CSV_files/estimated_filtrato.csv")
def movie_affinity(movies_id, user_df):
    """
    Calcola l'affinità di un film ai gusti di un utente basandosi sull'estimatedrating.

    :param user_id: ID dell'utente
    :param movie_id: ID del film da valutare
    :param df: DataFrame contenente i film e i loro estimatedrating
    :return: Affinità del film in percentuale
    """
    df_return = pd.DataFrame(columns=['movieId', 'affinity'])
    min_rating = user_df["estimatedrating"].min()
    max_rating = user_df["estimatedrating"].max()
    for movie_id in movies_id:
        movie_rating = user_df.loc[user_df["movieId"] == movie_id, "estimatedrating"].values[0]
        affinity_score = (movie_rating - min_rating) / (max_rating - min_rating) * 100
        row = {'movieId': movie_id, 'affinity': affinity_score}
        df_return.loc[len(df_return)] = row
    return df_return

def query_uri(uri):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>

        SELECT ?description WHERE {{
          <{uri}> dbo:abstract ?description .
          FILTER (lang(?description) = "it" || lang(?description) = "en")
        }}
        """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    descriptions = [result["description"]["value"] for result in results["results"]["bindings"]]
    description = descriptions[0] if descriptions else "Descrizione non trovata"
    print(description)

def query_dbpedia(imdb_id, film_name, movie_id):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
    SELECT ?film ?title ?imdbId WHERE {{

      # Controlliamo se esiste un film con l'IMDb ID specificato
      {{
        SELECT ?film ?title ?imdbId WHERE {{
          ?film rdf:type dbo:Film.
          ?film dbo:imdbId "{imdb_id}".  # Sostituisci con il tuo IMDb ID

          OPTIONAL {{ ?film rdfs:label ?title. }}
          OPTIONAL {{ ?film dbo:imdbId ?imdbId. }}
        }}
        LIMIT 1  # Se esiste, restituisce solo questo film
      }}

      UNION

      # Se non esiste un film con quell'IMDb ID, cerca per titolo
      {{
        SELECT ?film ?title ?imdbId WHERE {{
          ?film rdf:type dbo:Film.
          ?film rdfs:label ?title.
          OPTIONAL {{ ?film dbo:imdbId ?imdbId. }}

          # Cerca il titolo in modo case-insensitive e parziale
          FILTER(CONTAINS(LCASE(STR(?title)), "{film_name}"))  # Sostituisci con il tuo titolo
        }}
        ORDER BY STRLEN(?title)  # Prendi il titolo più conciso
        LIMIT 1  # Prendi solo un risultato
      }}
    }}
    LIMIT 1  # Assicura che venga restituito solo un risultato in totale
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    if results["results"]["bindings"] == []:
        return

    films = []
    for result in results["results"]["bindings"]:
        film_info = {
            "film": result.get("film", {}).get("value", "N/A"),
            "title": result.get("title", film_name),
            "imdb_id": result.get("imdb_id", imdb_id),
            "movie_id": result.get("movie_id", movie_id)
        }
        films.append(film_info)
        print(film_info)

    return films

def get_estimated():
    import pandas as pd

    # Carica i file CSV
    file1 = pd.read_csv("../CSV_files/ratings.csv")
    file2 = pd.read_csv("../CSV_files/estimated.csv")

    # Unisci i due DataFrame basandoti su userid e movieid per trovare le corrispondenze
    merged = file2.merge(file1[['userId', 'movieId']], on=['userId', 'movieId'], how='left', indicator=True)

    # Tieni solo le righe che NON sono presenti in file1
    file2_filtered = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Salva il risultato in un nuovo file
    file2_filtered.to_csv("../CSV_files/estimated_filtrato.csv", index=False)

    print("File filtrato salvato con successo!")


def top_ten(user):

    file = pd.read_csv("../CSV_files/mapping_ratings.csv", index_col=False, dtype={"imdbId": str})

    estimated_filtered = estimated[estimated["userId"] == user]
    top_10_votes = estimated_filtered.nlargest(10, "estimatedrating")
    merged = top_10_votes.merge(file, on="movieId", how="inner")
    movie_ids = merged['movieId'].tolist()
    df_affinity = movie_affinity(movie_ids, estimated_filtered)
    merged = merged.merge(df_affinity, on="movieId", how="inner")
    # Estrai la colonna "film" in una lista
    uris = merged["film"].tolist()

    uri_values = " ".join(f"<{uri}>" for uri in uris)


    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?uri ?title ?description (GROUP_CONCAT(DISTINCT ?actor; separator=", ") AS ?starring) WHERE {{
            VALUES ?uri {{ {uri_values} }}
            ?uri rdfs:label ?title .
            OPTIONAL {{ 
                ?uri dbo:abstract ?description .
                FILTER (lang(?description) = "en")
            }}
            OPTIONAL {{
                ?uri dbo:starring ?actor .
            }}
            FILTER (lang(?title) = "en")
        }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    films = []

    for result in results["results"]["bindings"]:
        uri = result["uri"]["value"]
        title = result["title"]["value"]
        description = result.get("description", {}).get("value", "No description available")
        actors = result["starring"]["value"]

        # Trova il rating corrispondente nel file CSV
        movie_rating = file.loc[file['film'] == uri, 'rating'].values
        rating = movie_rating[0] if len(movie_rating) > 0 else "N/A"

        # Trova l'imdbId e movieId corrispondente nel file CSV
        imdbId = file.loc[file['film'] == uri, 'imdb_id'].values
        movieId_array = file.loc[file['film'] == uri, 'movieId'].values
        imdbId = imdbId[0] if len(imdbId) > 0 else "N/A"
        movieId = movieId_array[0] if len(movieId_array) > 0 else None  # Estrai scalare


        if movieId is not None:
            # Assicurati che movieId sia dello stesso tipo di df_affinity['movieId']
            # Controlla il tipo di df_affinity['movieId'] (es. int o str)
            movieId = int(movieId)  # Converti a intero, se df_affinity['movieId'] è int
            # Oppure: movieId = str(movieId) se df_affinity['movieId'] è str

            # Filtra e ottieni affinity
            affinity_series = df_affinity[df_affinity['movieId'] == movieId]['affinity']

            if not affinity_series.empty:
                affinity = affinity_series.values[0]  # Estrai il valore scalare




        film_info = {
            "title": title,
            "description": description,
            "rating": round(float(rating), 1),
            "imdb_id": "tt" + imdbId,
            "actors": actors,
            "affinity": affinity
        }

        films.append(film_info)

    return films


def recommend_similar_movies(movie_id, similarity_matrix, not_seen_movies, top_n=3):
    # Verifica se movie_id è presente nella matrice di similarità
    if movie_id not in similarity_matrix.index:
        return "Film non trovato nel database!"

    # Ordina i film per similarità decrescente
    similar_movies = similarity_matrix[movie_id].sort_values(ascending=False)
    similar_movies_df = similar_movies.to_frame(name='similarity')

    # Inizializza una lista per accumulare i risultati
    result_list = []
    i = 0  # Contatore per i film trovati
    j = 0  # Contatore per iterare sui film simili

    # Cerca fino a trovare top_n film non visti
    while i < top_n and j < len(similar_movies_df):
        movie_id_j = similar_movies_df.index[j]
        find_row = not_seen_movies[not_seen_movies['movieId'] == movie_id_j]

        if not find_row.empty:
            # Aggiungi il movieId e la similarità alla lista dei risultati
            result_list.append({'movieId': movie_id_j, 'similarity': similar_movies_df.iloc[j]['similarity']})
            i += 1
        j += 1

    # Crea il DataFrame finale dai risultati accumulati
    movies_ret_df = pd.DataFrame(result_list, columns=['movieId', 'similarity'])

    # Se non ci sono risultati, restituisci un messaggio
    if movies_ret_df.empty:
        return "Nessun film simile trovato tra quelli non visti!"

    return movies_ret_df

def similar_movies(user):
    #estimated = pd.read_csv("../CSV_files/estimated_filtrato.csv")
    file = pd.read_csv("../CSV_files/mapping_ratings.csv", index_col=False, dtype={"imdbId": str})
    ratings_df = pd.read_csv("../CSV_files/ratings.csv", index_col=False)
    movies_df = pd.read_csv("../CSV_files/movies.csv", index_col=False)
    # Merge per avere il nome dei film associato alle valutazioni
    ratings_df = ratings_df.merge(movies_df, on='movieId')
    ratings_df = ratings_df.drop('timestamp', axis=1)
    # Creiamo una matrice film-utenti (pivot)
    movie_user_matrix = ratings_df.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
    # Calcoliamo la similarità coseno tra i film
    movie_similarity = cosine_similarity(movie_user_matrix)
    # Convertiamo in DataFrame per leggibilità
    movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)
    userid = user
    ratings_user = ratings_df[ratings_df['userId'] == userid]
    ratings_user = ratings_user.sort_values('rating', ascending=False)
    random_index = rd.randint(0, 9)
    movie = ratings_user.iloc[random_index]
    estimated_filtered = estimated
    rec_movies_id = recommend_similar_movies(movie['movieId'], movie_similarity_df, estimated_filtered, 10)
    movies_rec = rec_movies_id.merge(movies_df, on='movieId')
    merged = movies_rec.merge(file, on="movieId", how="inner")
    movie_ids = merged['movieId'].tolist()
    df_affinity = movie_affinity(movie_ids, estimated_filtered)
    merged = merged.merge(df_affinity, on="movieId", how="inner")
    print("Film simili a: ",movie['title'])
    print(merged)
    # Estrai la colonna "film" in una lista
    uris = merged["film"].tolist()

    uri_values = " ".join(f"<{uri}>" for uri in uris)


    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?uri ?title ?description (GROUP_CONCAT(DISTINCT ?actor; separator=", ") AS ?starring) WHERE {{
            VALUES ?uri {{ {uri_values} }}
            ?uri rdfs:label ?title .
            OPTIONAL {{ 
                ?uri dbo:abstract ?description .
                FILTER (lang(?description) = "en")
            }}
            OPTIONAL {{
                ?uri dbo:starring ?actor .
            }}
            FILTER (lang(?title) = "en")
        }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    films = []

    for result in results["results"]["bindings"]:
        uri = result["uri"]["value"]
        title = result["title"]["value"]
        description = result.get("description", {}).get("value", "No description available")
        actors = result["starring"]["value"]

        # Trova il rating corrispondente nel file CSV
        movie_rating = file.loc[file['film'] == uri, 'rating'].values
        rating = movie_rating[0] if len(movie_rating) > 0 else "N/A"

        # Trova l'imdbId e movieId corrispondente nel file CSV
        imdbId = file.loc[file['film'] == uri, 'imdb_id'].values
        movieId_array = file.loc[file['film'] == uri, 'movieId'].values
        imdbId = imdbId[0] if len(imdbId) > 0 else "N/A"
        movieId = movieId_array[0] if len(movieId_array) > 0 else None  # Estrai scalare


        if movieId is not None:
            # Assicurati che movieId sia dello stesso tipo di df_affinity['movieId']
            # Controlla il tipo di df_affinity['movieId'] (es. int o str)
            movieId = int(movieId)  # Converti a intero, se df_affinity['movieId'] è int
            # Oppure: movieId = str(movieId) se df_affinity['movieId'] è str

            # Filtra e ottieni affinity
            affinity_series = df_affinity[df_affinity['movieId'] == movieId]['affinity']

            if not affinity_series.empty:
                affinity = affinity_series.values[0]  # Estrai il valore scalare




        film_info = {
            "title": title,
            "description": description,
            "rating": round(float(rating), 1),
            "imdb_id": "tt" + imdbId,
            "actors": actors,
            "affinity": affinity,
            "movie_base": movie['title']
        }

        films.append(film_info)

    return films



def scopri_di_piu(user):
    #estimated = pd.read_csv("../CSV_files/estimated_filtrato.csv")
    file = pd.read_csv("../CSV_files/mapping_ratings.csv", index_col=False, dtype={"imdbId": str})

    est = estimated[estimated["userId"] == user]
    # Se la colonna "n_selezioni" non esiste, la creiamo con valore iniziale 0
    if "n_selezioni" not in est.columns:
        est["n_selezioni"] = 0
    df_non_visti = est.copy()

    # Numero totale di round di selezione
    k = 5000
    t = 1  # Numero totale di selezioni effettuate (iniziamo da 1 per evitare log(0))
    # Pre-calcoliamo il log(t) per evitare di ricalcolarlo ogni volta
    log_t = np.log(np.arange(1, k * 10 + 1))
    l = 4
    for i in range(k):
        # Calcolare UCB solo per i film non visti
        ucb_values = df_non_visti["estimatedrating"] + np.sqrt((l * log_t[t - 1]) / (df_non_visti["n_selezioni"] + 1))

        # Selezionare i 10 film con UCB più alto
        top_10_indices = np.argpartition(-ucb_values, 10)[:10]  # Più efficiente di `nlargest`

        # Aggiornare il conteggio delle selezioni per i film scelti
        df_non_visti.iloc[top_10_indices, df_non_visti.columns.get_loc("n_selezioni")] += 1

        # Aggiornare il numero totale di selezioni fatte
        t += 10
    final_ucb_values = df_non_visti["estimatedrating"] + np.sqrt((l * np.log(t)) / (df_non_visti["n_selezioni"] + 1))
    df_non_visti["UCB"] = final_ucb_values

    # Selezionare i 10 migliori film finali
    top_10_votes = df_non_visti.nlargest(10, "UCB")
    merged = top_10_votes.merge(file, on="movieId", how="inner")
    movie_ids = merged['movieId'].tolist()
    df_affinity = movie_affinity(movie_ids, est)
    merged = merged.merge(df_affinity, on="movieId", how="inner")

    # Estrai la colonna "film" in una lista
    uris = merged["film"].tolist()

    uri_values = " ".join(f"<{uri}>" for uri in uris)


    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?uri ?title ?description (GROUP_CONCAT(DISTINCT ?actor; separator=", ") AS ?starring) WHERE {{
            VALUES ?uri {{ {uri_values} }}
            ?uri rdfs:label ?title .
            OPTIONAL {{ 
                ?uri dbo:abstract ?description .
                FILTER (lang(?description) = "en")
            }}
            OPTIONAL {{
                ?uri dbo:starring ?actor .
            }}
            FILTER (lang(?title) = "en")
        }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    films = []

    for result in results["results"]["bindings"]:
        uri = result["uri"]["value"]
        title = result["title"]["value"]
        description = result.get("description", {}).get("value", "No description available")
        actors = result["starring"]["value"]

        # Trova il rating corrispondente nel file CSV
        movie_rating = file.loc[file['film'] == uri, 'rating'].values
        rating = movie_rating[0] if len(movie_rating) > 0 else "N/A"

        # Trova l'imdbId e movieId corrispondente nel file CSV
        imdbId = file.loc[file['film'] == uri, 'imdb_id'].values
        movieId_array = file.loc[file['film'] == uri, 'movieId'].values
        imdbId = imdbId[0] if len(imdbId) > 0 else "N/A"
        movieId = movieId_array[0] if len(movieId_array) > 0 else None  # Estrai scalare

        if movieId is not None:
            # Assicurati che movieId sia dello stesso tipo di df_affinity['movieId']
            # Controlla il tipo di df_affinity['movieId'] (es. int o str)
            movieId = int(movieId)  # Converti a intero, se df_affinity['movieId'] è int
            # Oppure: movieId = str(movieId) se df_affinity['movieId'] è str

            # Filtra e ottieni affinity
            affinity_series = df_affinity[df_affinity['movieId'] == movieId]['affinity']
            if not affinity_series.empty:
                affinity = affinity_series.values[0]  # Estrai il valore scalare
                print(f"Affinità per il film {title}: {affinity}")



        film_info = {
            "title": title,
            "description": description,
            "rating": round(float(rating), 1),
            "imdb_id": "tt" + imdbId,
            "actors": actors,
            "affinity": affinity
        }

        films.append(film_info)

    return films


def run():
    '''estimated = pd.read_csv("../CSV_files/estimated.csv", index_col=False)
    estimated_filtered = estimated[estimated["userId"] == 1000]
    max_vote = estimated_filtered.loc[[estimated_filtered["estimatedrating"].idxmax()]]
    print(max_vote)

    links = pd.read_csv("../CSV_files/links.csv", index_col=False, dtype={"imdbId": str})
    links_filtered = links[links["movieId"] == max_vote["movieId"].values[0]]
    imdb_id = links_filtered["imdbId"].values[0]
    print(imdb_id)
    imdb_id = '1375666'
    film_name = 'Call Me Bwana'
    query_dbpedia(imdb_id, film_name)'''

    '''
    ---CASO 2
    estimated = pd.read_csv("../CSV_files/estimated.csv", index_col=False)
    estimated_filtered = estimated[estimated["userId"] == 1]
    max_vote = estimated_filtered.loc[[estimated_filtered["estimatedrating"].idxmax()]]
    print(max_vote)

    file = pd.read_csv("../CSV_files/mapping.csv", index_col=False, dtype={"imdbId": str})

    movie = file.loc[file["movie_id"] == max_vote.iloc[0]["movieId"]]
    uri = movie.iloc[0]["film"]
    query_uri(uri)

    get_estimated()


    films = []'''


    get_actor_abstract("aa")


    '''for imdb_id, title, movie_id in zip(imdb_ids, title, movie_ids):
        res = query_dbpedia(imdb_id, title, movie_id)
        films.append(res)

    df = pd.DataFrame(films)
    df.to_csv("../CSV_files/mapping.csv", index=False, encoding="utf-8")'''




if __name__ == '__main__':
    run()
