import numpy as np
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import Connect_to_DBPedia as cb

def calculate_user_actor(userid):
    df = pd.read_csv("../CSV_files/ratings_actor.csv", index_col=False)
    print("DF:\n", df)

    user_df = df[df["userId"] == userid]
    max_row = user_df.loc[user_df["rating"].idxmax()]

    return max_row["actor"]
    print("Attore:", actor)



def get_films_by_actor(user_id):

    df = pd.read_csv("../CSV_files/ratings_actor.csv", index_col=False)
    actor = calculate_user_actor(user_id)

    print("Attore:", actor)

    ratings = pd.read_csv("../CSV_files/mapping_ratings.csv", index_col=False, dtype={"imdbId": str})
    films = df[(df["actor"].str.contains(actor, case=False, na=False))]["movieId"].unique()

    estimated = pd.read_csv("../CSV_files/estimated_filtrato.csv", index_col=False)
    estimated = estimated[(estimated["userId"] == user_id) & (estimated["movieId"].isin(films))]
    estimated_ordinato = estimated.sort_values(by="estimatedrating", ascending=False)

    top_films_id = estimated_ordinato["movieId"].head(10).tolist()


    df_affinity = cb.movie_affinity(top_films_id, estimated)

    top_films = ratings[(ratings["movieId"].isin(top_films_id))]["film"].tolist()
    print("Estimated:\n", top_films)



    uri_values = " ".join(f"<{uri}>" for uri in top_films)

    # effettuo la query
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
        movie_rating = ratings.loc[ratings['film'] == uri, 'rating'].values
        rating = movie_rating[0] if len(movie_rating) > 0 else "N/A"

        # Trova l'imdbId corrispondente nel file CSV
        imdbId = ratings.loc[ratings['film'] == uri, 'imdb_id'].values
        imdbId = imdbId[0] if len(imdbId) > 0 else "N/A"
        movieId_array = ratings.loc[ratings['film'] == uri, 'movieId'].values
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
            "actor": actor,
            "actors": actors,
            "affinity": affinity
        }

        films.append(film_info)

    return films



def get_actor_info(user_id, actor):
    df = pd.read_csv("../CSV_files/ratings_actor.csv", index_col=False)
    #actor = calculate_user_actor(user_id)

    print("Attore:", actor)

    ratings = pd.read_csv("../CSV_files/mapping_ratings.csv", index_col=False, dtype={"imdbId": str})
    films = df[(df["actor"].str.contains(actor, case=False, na=False))]["movieId"].unique()

    estimated = pd.read_csv("../CSV_files/estimated_filtrato.csv", index_col=False)
    estimated = estimated[(estimated["userId"] == user_id) & (estimated["movieId"].isin(films))]
    estimated_ordinato = estimated.sort_values(by="estimatedrating", ascending=False)

    top_films_id = estimated_ordinato["movieId"].head(10).tolist()

    top_films = ratings[(ratings["movieId"].isin(top_films_id))]["film"].tolist()
    print("Estimated:\n", top_films)


    # effettuo la query
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    print(actor)
    actor = actor.replace(" ", "_")
    actor_uri = "http://dbpedia.org/resource/" + actor
    print(actor_uri)

    query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?comment ?image WHERE {{
                <{actor_uri}> rdfs:comment ?comment .
                OPTIONAL {{
                    <{actor_uri}> dbo:thumbnail ?image
                }}
                FILTER (lang(?comment) = "en")
            }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    comment = results["results"]["bindings"][0]["comment"]["value"]
    image = results["results"]["bindings"][0]["image"]["value"]

    films = []

    for film in top_films:
        # Trova l'imdbId corrispondente nel file CSV
        imdbId = ratings.loc[ratings['film'] == film, 'imdb_id'].values
        imdbId = imdbId[0] if len(imdbId) > 0 else "N/A"
        movie = {
            "film": film,
            "imdb_id": imdbId
        }
        films.append(movie)

    actor_info = {
        "movies": films,
        "comment": comment,
        "image": image
    }

    return actor_info






if __name__ == '__main__':
    get_actor_info(1, "Tom Hanks")