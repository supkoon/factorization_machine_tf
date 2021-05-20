
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class dataloader:
    def __init__(self,datapath):
        self.datapath =datapath
        ratings_df = pd.read_csv(os.path.join(self.datapath,"ratings.csv"),encoding='utf-8')
        ratings_df.drop('timestamp', inplace=True, axis=1)

        movies_df = pd.read_csv(os.path.join(self.datapath, "movies.csv"), encoding='utf-8', index_col='movieId')
        genre_df = movies_df['genres'].str.get_dummies(sep='|')
        movies_df = pd.concat([movies_df, genre_df], axis=1)
        movies_df.drop("genres", inplace=True, axis=1)
        movies_df['year'] = movies_df["title"].str.extract('(\(\d{4}\))')
        movies_df['year'] = movies_df['year'].apply(lambda x: str(x).replace('(', '').replace(')', ""))
        movies_df.drop('title', axis=1, inplace=True)
        movies_df = movies_df.reset_index()

        feature_vector = pd.merge(ratings_df, movies_df, how="inner", on="movieId")

        user_onehot = pd.get_dummies(feature_vector['userId'], prefix='user')
        item_onehot = pd.get_dummies(feature_vector['movieId'], prefix='movie')

        concat_feature_vector = pd.concat([feature_vector, user_onehot, item_onehot], axis=1).drop("userId",axis=1).drop( "movieId", axis=1)
        concat_feature_vector['year'] = concat_feature_vector['year'].astype('float32')

        target_rating = concat_feature_vector["rating"]
        concat_feature_vector.drop('rating', axis=1, inplace=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(concat_feature_vector, target_rating, test_size=0.1)

    def generate_trainset(self):
        return self.X_train,self.y_train

    def generate_testset(self):
        return self.X_test,self.y_test





