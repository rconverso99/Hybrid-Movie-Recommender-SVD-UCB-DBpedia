import numpy as np
import pandas as pd
from surprise import SVD
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from SPARQLWrapper import SPARQLWrapper, JSON
import Connect_to_DBPedia as cb


def manage_data(df):
    # Import data into a DataFrame and drop unnecessary columns
    df2 = df[['userId', 'genres', 'rating']]

    # Instantiate reader and data
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df2, reader)

    # Train test split with test size of 20%
    trainset, testset = train_test_split(data, test_size=.2)

    # Print number of users and items for the trainset
    print('Number of users in train set : ', trainset.n_users, '\n')
    print('Number of items in train set : ', trainset.n_items, '\n')

    return trainset, testset, data

def SVD_model(trainset, testset, data):
    # Reinstantiate the model with the best parameters from GridSearch
    svdtuned = SVD(n_factors=80,
                   reg_all=0.06,
                   n_epochs=30,
                   lr_all=0.01)

    # Fit and predict the model
    svdtuned.fit(trainset)
    svdpreds = svdtuned.test(testset)

    # Print RMSE and MAE results
    accuracy.rmse(svdpreds)
    accuracy.mae(svdpreds)

    # Perform 3-Fold cross-validation for SVD tuned model
    cv_svd_tuned = cross_validate(svdtuned, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    for i in cv_svd_tuned.items():
        print(i)
    np.mean(cv_svd_tuned['test_rmse'])

def train_new_model(new_ratings_df):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(new_ratings_df, reader)
    trainset, testset = train_test_split(data, test_size=.2)

    # Reinstantiate the model with the best parameters from GridSearch and fit on the trainset
    svdtuned2 = SVD(n_factors=80,
                    reg_all=0.06,
                    n_epochs=30,
                    lr_all=0.01)
    svdtuned2.fit(trainset)

    # Find predictions for the three movies that user with userId=1000 just rated
    print(svdtuned2.predict(1000, 1240))
    print(svdtuned2.predict(1000, 96610))
    print(svdtuned2.predict(1000, 6534))

    return svdtuned2

def extract_prediction(new_ratings_df, svdtuned2):
    # Split genres into individual genres using explode()
    new_ratings_df['genres'] = new_ratings_df['genres'].str.split('|')
    new_ratings_df = new_ratings_df.explode('genres')

    # Create list of unique userIds and genres
    userids = new_ratings_df['userId'].unique()
    genres = new_ratings_df['genres'].unique()

    # Create a list and append the userId, genre, and estimated ratings
    predictions = []
    for u in userids:
        for g in genres:
            predicted = svdtuned2.predict(u, g)  # Predict rating for user u and genre g
            predictions.append([u, g, predicted[3]])  # Store the result in the list

    # Convert the list to a dataframe
    estimated = pd.DataFrame(predictions)
    estimated.rename(columns={0: 'userId', 1: 'genres', 2: 'estimatedrating'}, inplace=True)
    print("Estimated:")
    print(estimated)
    estimated.to_csv('../CSV_files/estimated_genres_single.csv')

def calulate_genre(user_id):
    df = pd.read_csv("../CSV_files/estimated_genres_single.csv", index_col=False)
    user_df = df[df["userId"] == user_id]
    max_row = user_df.loc[user_df["rating"].idxmax()]

    return max_row["genres"]


def get_top_film_genre(user_id):
    '''file = calulate_genre(user_id)
    print("File:\n", file)
    genre = file[file["userId"] == user_id]["preferred_genre"].iloc[0]'''
    genre = calulate_genre(user_id)
    #genre = genre["preferred_genre"].iloc[0]
    print(genre)
    mapping = pd.read_csv("../CSV_files/mapping_ratings.csv", index_col=False)
    movies = pd.read_csv("../CSV_files/movies.csv", index_col=False)
    ratings = pd.read_csv("../CSV_files/mapping_ratings.csv", index_col=False, dtype={"imdbId": str})
    estimated = pd.read_csv("../CSV_files/estimated_filtrato.csv", index_col=False)
    estimated = estimated[(estimated["userId"] == user_id)]

    #prendo i film con il genre presente in genre
    filtered_movies = movies[movies["genres"].str.contains(genre, case=False, na=False)]
    movie_ids = filtered_movies["movieId"].tolist()

    # prendo il campo "film" dei 10 film con il genre presente in genre e con il rating più alto
    filtered_df = estimated[estimated["movieId"].isin(movie_ids)]
    sorted_df = filtered_df.sort_values(by="estimatedrating", ascending=False)
    top_films_id = sorted_df["movieId"].head(10).tolist()
    top_films = []
    for movieId in top_films_id:
        film_values = ratings.loc[ratings["movieId"] == movieId, "film"].values
        if len(film_values) > 0:  # Controlla se ci sono risultati
            top_films.append(film_values[0])  # Prendi il primo valore
    #ratings.loc[ratings['film'] == uri, 'imdb_id'].values
    #top_films = sorted_df["film"].head(10).tolist()
    #estimated_ordinato = estimated.sort_values(by="estimatedrating", ascending=False)
    df_affinity = cb.movie_affinity(top_films_id, estimated)
    print("Top Film:\n", top_films)
    uri_values = " ".join(f"<{uri}>" for uri in top_films)

    #effettuo la query
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
            GROUP BY ?uri ?title ?description
        """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    print(results)
    films = []

    for result in results["results"]["bindings"]:
        uri = result["uri"]["value"]
        title = result["title"]["value"]
        actors = result["starring"]["value"]
        description = result.get("description", {}).get("value", "No description available")

        # Trova il rating corrispondente nel file CSV
        movie_rating = ratings.loc[ratings['film'] == uri, 'rating'].values
        rating = movie_rating[0] if len(movie_rating) > 0 else "N/A"

        # Trova l'imdbId corrispondente nel file CSV
        imdbId = ratings.loc[ratings['film'] == uri, 'imdb_id'].values
        imdbId = imdbId[0] if len(imdbId) > 0 else "N/A"
        movieId_array = ratings.loc[ratings['film'] == uri , 'movieId'].values
        movieId = movieId_array[0] if len(movieId_array) > 0 else None  # Estrai scalare

        if movieId is not None:
            # Assicurati che movieId sia dello stesso tipo di df_affinity['movieId']
            # Controlla il tipo di df_affinity['movieId'] (es. int o str)
            movieId = int(movieId)  # Converti a intero, se df_affinity['movieId'] è int
            # Oppure: movieId = str(movieId) se df_affinity['movieId'] è str

            # Filtra e ottieni affinity
            affinity_series = df_affinity[df_affinity['movieId'] == movieId]['affinity']
            # print(df_affinity)
            if not affinity_series.empty:
                affinity = affinity_series.values[0]  # Estrai il valore scalare

        film_info = {
            "title": title,
            "description": description,
            "rating": round(float(rating), 1),
            "imdb_id": "tt" + imdbId,
            "genre": genre,
            "actors": actors,
            "affinity": affinity
        }

        films.append(film_info)

    print("Result:", films)

    return films






def run():
    '''ratings = pd.read_csv("../CSV_files/ratings.csv", index_col=False)
    movies = pd.read_csv("../CSV_files/movies.csv", index_col=False)
    df = pd.merge(ratings, movies, on='movieId', how='left')
    df2 = df[['userId', 'genres', 'rating']]

    train, test, data = manage_data(df)
    SVD_model(train, test, data)

    dfnew = df[['userId', 'genres', 'rating']]
    svdtuned2 = train_new_model(df2)
    extract_prediction(df2, svdtuned2)'''

    # Load the estimated ratings
    get_top_film_genre(1)

if __name__ == '__main__':
    run()
