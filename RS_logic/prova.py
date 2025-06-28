import pandas as pd
import re
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from bs4 import BeautifulSoup

def add_ratings():
    rating = pd.read_csv("../CSV_files/ratings.csv")
    mapping = pd.read_csv("../CSV_files/mapping.csv")

    media_rating = rating.groupby("movieId")["rating"].mean().reset_index()
    file = mapping.merge(media_rating, on="movieId", how="left")
    print(file)

    file.to_csv("../CSV_files/mapping_ratings.csv", index=False)

def film():
    movies = pd.read_csv("../CSV_files/movies.csv")
    movies = movies[['movieId', 'genres']]
    mapping = pd.read_csv("../CSV_files/mapping_ratings.csv")
    file = mapping.merge(movies, on="movieId", how="left")
    print(file.iloc[0])
    file = file[['film', 'title', 'imdb_id', 'movieId', 'rating', 'genres']]
    file.to_csv("../CSV_files/film.csv", index=False)





if __name__ == '__main__':
    film()