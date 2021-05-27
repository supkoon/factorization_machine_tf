
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

class dataloader:
    def __init__(self,datapath):
        self.datapath =datapath
        ratings_df = pd.read_csv(os.path.join(self.datapath,"ratings.csv"),encoding='utf-8')
        ratings_df.drop('timestamp', inplace=True, axis=1)

        movies_df = pd.read_csv(os.path.join(self.datapath, "movies.csv"), encoding='utf-8')
        movies_df = movies_df.set_index("movieId")
        dummy_genre_df =  movies_df['genres'].str.get_dummies(sep='|')


        movies_df['year'] = movies_df["title"].str.extract('(\(\d\d\d\d\))')
        movies_df['year'] = movies_df['year'].astype('str')
        movies_df['year'] = movies_df['year'].map(lambda x: x.replace("(", "").replace(")", ""))
        movies_df['year'] = movies_df['year'].astype("float32").astype("int32")
        movies_df.drop(movies_df[movies_df['year'] == 0].index, inplace=True, axis=0)
        movies_df.drop('title',axis=1,inplace=True)
        bins = list(range(1900, 2021, 20))
        labels = [x for x in range(len(bins) - 1)]
        movies_df['year_level'] = pd.cut(movies_df['year'], bins, right=False, labels=labels)
        movies_df.drop('year', inplace=True, axis=1)


        threshold = 10
        over_threshold = ratings_df.groupby('movieId').size() >= threshold
        ratings_df['over_threshold'] = ratings_df['movieId'].map(lambda x: over_threshold[x])
        ratings_df = ratings_df[ratings_df["over_threshold"] == True]
        ratings_df.drop("over_threshold", axis=1, inplace=True)

        random_idx = np.random.permutation(len(ratings_df))
        shuffled_df = ratings_df.iloc[random_idx]

        concat_df = pd.concat([
            pd.get_dummies(shuffled_df['userId'], prefix="user"),
            pd.get_dummies(shuffled_df['movieId'], prefix="movie"),
            shuffled_df['movieId'].apply(lambda x: dummy_genre_df.loc[x]),
            shuffled_df['movieId'].apply(lambda x: movies_df.loc[x]["year_level"]).rename('year_level'),
        ], axis=1)

        target_df = ratings_df.loc[concat_df.index]['rating']
        target_df = target_df.apply(lambda x: 1 if x >= 4 else 0)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(concat_df, target_df, test_size=0.1)

if __name__ == "__main__":
    print(print("---dataloader---"))



