import torch
from torch.utils.data import Dataset
import pandas as pd

class MovieRatingDataset(Dataset):
    def __init__(self, df, movieId2idx, idx2movieId, userId2idx=None,  idx2userId=None):
        self.df = df.copy()
        self.movieId2idx = movieId2idx
        # self.userId2idx = userId2idx
        # self.idx2userId = idx2userId
        self.idx2movieId = idx2movieId
        
        self.num_users = len(userId2idx)
        self.num_movies = len(movieId2idx)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = torch.tensor(self.df.iloc[idx]['userId'], dtype=torch.int32)
        movie_id = torch.tensor(self.df.iloc[idx]['movieId'], dtype=torch.int32)
        rating = torch.tensor(self.df.iloc[idx]['rating'], dtype=torch.float32)
        return user_id, movie_id, rating

def load_datasets(data_config):
    # Load data from CSV files
    # movie_df = pd.read_csv(data_config['movie_data_path'])
    # rating_df = pd.read_csv(data_config['rating_data_path'])
    new_movie_df = pd.read_csv(data_config['new_movie_df_path'])
    new_user_df = pd.read_csv(data_config['new_user_df_path'])
    
    # Create movieId2idx and userId2idx dictionaries
    movieId2idx = {movieId: idx for idx, movieId in enumerate(new_movie_df['movieId'])}
    userId2idx = {userId: idx for idx, userId in enumerate(new_user_df['userId'])}

    # Create reverse mappings for idx to user and idx to movies
    idx2userId = {idx: userId for userId, idx in userId2idx.items()}
    idx2movieId = {idx: movieId for movieId, idx in movieId2idx.items()}
    
    # Map the original movie and user IDs to their corresponding indices
    new_user_df['movieId'] = new_user_df['movieId'].map(movieId2idx)
    new_user_df['userId'] = new_user_df['userId'].map(userId2idx)

    
    return MovieRatingDataset(new_user_df, movieId2idx, idx2movieId, userId2idx, idx2userId)