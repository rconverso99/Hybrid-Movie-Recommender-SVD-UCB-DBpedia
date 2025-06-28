#Import necessary libraries
import numpy as np
import pandas as pd
from surprise import SVD
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

def manage_data(df):
    # Import data into a DataFrame and drop unnecessary columns

    df2 = df[['userId', 'movieId', 'rating']]

    # Instansiate reader and data
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df2, reader)

    # Train test split with test sizre of 20%
    trainset, testset = train_test_split(data, test_size=.2)

    # Print number of uses and items for the trainset
    print('Number of users in train set : ', trainset.n_users, '\n')
    print('Number of items in train set : ', trainset.n_items, '\n')

    return trainset, testset, data


def SVD_model(trainset, testset, data):
    #Find best parameters
    '''parameters = {'n_factors': [20, 50, 80],
                  'reg_all': [0.04, 0.06],
                  'n_epochs': [10, 20, 30],
                  'lr_all': [.002, .005, .01]}
    gridsvd = GridSearchCV(SVD, param_grid=parameters, n_jobs=-1)

    # Fit SVD model on data
    gridsvd.fit(data)

    # Print best score and best parameters from the GridSearch
    print(gridsvd.best_score)
    print(gridsvd.best_params)'''

    # Reinstantiate the model with the best parameters fromGridSearch
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

    # Perform 3-Fold cross validation for SVD tuned model
    cv_svd_tuned = cross_validate(svdtuned, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    # Display the results for all 3-folds
    for i in cv_svd_tuned.items():
        print(i)
    # Print out the average RMSE score for the test set
    np.mean(cv_svd_tuned['test_rmse'])


def generating_new_ratings(movie_df, num, genre=None, user_id = 1000):
    # Create an empty list of ratings
    rating_list = []

    # For all number of ratings, provide a random movie sample within the specified genre for the user to rate
    while num > 0:
        if genre:
            movie = movie_df[movie_df['genres'].str.contains(genre)].sample(1)
        else:
            movie = movie_df.sample(1)
        print(movie["title"])

        # Provide user with a prompt to rate the movie, then print the userID, movieID, then title, then append
        # results to the rating_list
        rating = input('How do you rate this movie on a scale of 1-5, press n if you have not seen :\n')
        if rating == 'n':
            continue
        else:
            rating_one_movie = {'userId': user_id, 'movieId': movie['movieId'].values[0],
                                'title': movie['title'].values[0], 'rating': rating}
            rating_list.append(rating_one_movie)
            num -= 1
    return rating_list


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
    # Create list of unique userIds and movieIds
    userids = new_ratings_df['userId'].unique()
    movieids = new_ratings_df['movieId'].unique()

    # Create a list and append the userId, movieId, and estimated ratings
    predictions = []
    for u in userids:
        for m in movieids:
            predicted = svdtuned2.predict(u, m)
            predictions.append([u, m, predicted[3]])

    # Convert the list to a dataframe
    estimated = pd.DataFrame(predictions)
    # rename columns of DataFrame
    estimated.rename(columns={0: 'userId', 1: 'movieId', 2: 'estimatedrating'}, inplace=True)
    print("Extimated:")
    print(estimated)
    estimated.to_csv('../CSV_files/estimated.csv')



def run():
    ratings = pd.read_csv("../CSV_files/ratings.csv", index_col=False)
    movies = pd.read_csv("../CSV_files/movies.csv", index_col=False)
    df = pd.merge(ratings, movies, on='movieId', how='left')
    df2 = df[['userId', 'movieId', 'rating']]

    train, test, data = manage_data(df)
    #SVD_model(train, test, data)

    dfnew = df[['userId', 'movieId', 'rating', 'title', 'genres']]
    userrating = generating_new_ratings(dfnew, 3, 'Action')
    userrating = pd.DataFrame(userrating)
    print(userrating)

    # Add new ratings to our DataFrame
    new_ratings_df = pd.concat([df2, userrating], ignore_index=True, sort=False)
    # Drop the 'title' column so that our dataframe is ready to be put into surprise
    new_ratings_df.drop(['title'], axis=1, inplace=True)
    print(new_ratings_df)

    svdtuned2 = train_new_model(new_ratings_df)
    extract_prediction(new_ratings_df, svdtuned2)



if __name__ == '__main__':
    run()

