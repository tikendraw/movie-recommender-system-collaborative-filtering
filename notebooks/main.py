# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

import pandas as pd
import numpy as np

import time
from pathlib import Path

cur_dir = Path('.').absolute()
data_dir=cur_dir.parent/ 'data'

# Exploring Movie df
movie_df = pd.read_csv(data_dir/'movie.csv')

def get_movie_name(idx, df):
    try:
        return df[df.movieId==idx].title.values[0]
    except IndexError as e:
        print("IndexError:", idx)

def get_movie_id(movie_name, df):
    try:
        return df[df.title==movie_name].movieId.values[0]
    except IndexError as e:
        print("Movie not found in dataset")

movie_df.isna().sum() # no nulls

movie_df.duplicated().sum() #no duplicates

# Calculate the value counts for each movie title
title_value_counts = movie_df['title'].value_counts()

# Filter titles that appear more than once
duplicate_titles = title_value_counts[title_value_counts > 1].index.tolist()

all_genres = movie_df.genres.apply(lambda x : ' '.join(str(x).split('|'))).values.tolist() # split from |
all_genres = ' '.join(set(all_genres)).split() # join all strings and break them into words
all_genres = set(all_genres)  # make a set to find unique ones

# Exploraing User data
user_df = pd.read_csv(data_dir/'rating.csv', usecols=['userId','movieId','rating'])

# this columns are using too much precision for very low values, lowering the datatype precision
user_df['movieId'] = user_df['movieId'].astype('int32') # dont lower too much as it changes the numbers to accomodate to the range
user_df['userId'] = user_df['userId'].astype('int32')
user_df['rating'] = user_df['rating'].astype('float32')


# finding out reviews per movies
movie_vote_count = user_df.movieId.value_counts()
movie_df_rating_filter = 100
popular_movieIds = movie_vote_count[movie_vote_count>=movie_df_rating_filter].index
new_user_df = user_df[user_df['movieId'].isin(popular_movieIds)]

# same for the users, if users with certain number of reviews  will be used.
user_vote_count = user_df.userId.value_counts()
user_df_vote_filter = 500
popular_userIds = user_vote_count[user_vote_count>=user_df_vote_filter].index
new_user_df = new_user_df[new_user_df['userId'].isin(popular_userIds)]
new_movie_df = movie_df[movie_df['movieId'].isin(popular_movieIds)]

# save dataframes
new_movie_df.to_csv(data_dir/'new_movie_df.csv')
new_user_df.to_csv(data_dir/'new_user_df.csv')

# dataset

# Define a custom dataset class
class MovieRatingDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()
        
        users = df.userId.sort_values().unique()
        movies = df.movieId.sort_values().unique()
        
        self.num_users = len(users)
        self.num_movies = len(movies) 
        
        self.userId2idx = {userId:idx for idx, userId in enumerate(users)}
        self.movieId2idx = {movieId:idx for idx, movieId in enumerate(movies)}
        
        self.idx2userId = {idx:userId for userId, idx in self.userId2idx.items()}
        self.idx2movieId = {idx:movieId for movieId, idx in self.movieId2idx.items()}
        
        self.df.movieId =  self.df.movieId.map(self.movieId2idx)
        self.df.userId =  self.df.userId.map(self.userId2idx)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        user_id = torch.tensor(self.df.iloc[idx]['userId'], dtype=torch.int32)
        movie_id = torch.tensor(self.df.iloc[idx]['movieId'], dtype=torch.int32)
        rating = torch.tensor(self.df.iloc[idx]['rating'], dtype=torch.float32)
        return user_id, movie_id, rating


# Model
class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, model_path:Path=None):
        super(RecommenderModel, self).__init__()
        self.model_path = model_path
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.user_embedding_bias = nn.Embedding(num_users, 1)
        self.movie_embedding_bias = nn.Embedding(num_movies, 1)
        self.out = nn.Linear(embedding_dim, 1)
        
        
        self.user_embedding.weight.data.uniform_(0, 0.05)
        self.movie_embedding.weight.data.uniform_(0, 0.05)
        self.user_embedding_bias.weight.data.uniform_(-0.01, 0.01)
        self.movie_embedding_bias.weight.data.uniform_(-0.01, 0.01)
        
    def forward(self, user_ids, movie_tags, debug=False):
       
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_tags)

        user_emb_bias = self.user_embedding_bias(user_ids)
        movie_emb_bias = self.movie_embedding_bias(movie_tags)

        interaction = (user_emb * movie_emb) + user_emb_bias + movie_emb_bias
        return self.out(interaction)
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        
        try:
            self.load_state_dict(torch.load(model_path))
            print('Model weights loaded.')
        except FileNotFoundError as e:
            print(f'Weights not found. {e}')
        except RuntimeError as e:
            raise(e)
            
            
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        torch.save(self.state_dict(), model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, dataloader, optimizer, loss_function, num_epochs=10, device='cpu', data_percent=1.0, steps_per_epoch=None):
    model.to(device)
    print(f'{model.__class__.__name__} Running on: {device}')

    data_size = int(data_percent * len(dataloader))
    dataloader = iter(dataloader)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        epoch_progress = tqdm(range(data_size), desc=f"Epoch [{epoch+1:2}/{num_epochs:2}]")
        
        if steps_per_epoch is not None:
            epoch_progress = tqdm(range(steps_per_epoch), desc=f"Epoch [{epoch+1:2}/{num_epochs:2}]")

        last_update_time = time.time() - 1.0  # Initialize to ensure the first update
        
        for _ in epoch_progress:
            try:
                batch = next(dataloader)
            except StopIteration:
                print("Dataloader is exhausted. Resetting or stopping training.")
                break

            user_ids, movie_ids, ratings = batch

            
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            
            outputs = model(user_ids, movie_ids).squeeze()
            
            
            loss = loss_function(outputs, ratings)
            mse = F.mse_loss(outputs, ratings)
            mae = F.l1_loss(outputs, ratings)
            
            loss.backward()
            optimizer.step()
            
            total_mse += mse.item()
            total_mae += mae.item()
            total_samples += len(ratings)
            total_loss += loss.item()

            formatted_loss = f"{loss.item():.8f}"
            formatted_mse = f"{mse.item():.8f}"
            formatted_mae = f"{mae.item():.8f}"
            
            current_time = time.time()
            if current_time - last_update_time > epoch_progress.mininterval:
                epoch_progress.set_postfix({"Loss": formatted_loss, "MSE": formatted_mse, "MAE": formatted_mae})
                epoch_progress.update()
                last_update_time = current_time

            if steps_per_epoch is not None and _ + 1 >= steps_per_epoch:
                break

        # epoch_progress.close()
        average_loss = total_loss / min(data_size, steps_per_epoch) if steps_per_epoch is not None else total_loss / data_size
        average_mse = total_mse / min(data_size, steps_per_epoch) if steps_per_epoch is not None else total_mse / data_size
        average_mae = total_mae / min(data_size, steps_per_epoch) if steps_per_epoch is not None else total_mae / data_size
        
        print(f"Epoch [{epoch+1:2}/{num_epochs:2}] - Average Loss: {average_loss:.8f} - Average MSE: {average_mse:.8f} - Average MAE: {average_mae:.8f}")
        print()

# training
batch_size = 32

dataset = MovieRatingDataset(new_user_df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize your model, optimizer, and loss function
num_users = dataset.num_users
num_movies = dataset.num_movies
dim = 8

model = RecommenderModel(num_users, num_movies, dim)
optimizer = optim.AdamW(model.parameters(), lr=0.0015)
loss_function = nn.MSELoss()

# set model path
model_dir = cur_dir.parent/'models'
model_path = model_dir/'model.pth'
model.model_path=model_path

# load the model is exists
# model.load_model()

num_epochs = 10

train_model(model, dataloader,  optimizer, loss_function, num_epochs=num_epochs, device=device, data_percent=0.1)

# save the model
model.save_model()

# Getting recommendations

trained_movie_embedding = model.movie_embedding.weight.data.cpu().numpy()
trained_movie_embedding.shape

from sklearn.cluster import KMeans

clusters = 10
kmeans = KMeans(n_clusters=clusters,random_state=0).fit(trained_movie_embedding)

for cluster in range(clusters):
    print('Cluster: ',cluster)
    movs = []
    
    for movidx in np.where(kmeans.labels_==cluster)[0]:
        # print(movidx)
        movieid = dataset.idx2movieId[movidx]
        movie_title = movie_df[movie_df.movieId==movieid].title.values
        movs.append(movie_title)
        print('\t',movie_title)
        
        if len(movs)==15:
            break

def find_similar_movies(target_movie_embedding, all_movie_embeddings, top_n=5):
    with torch.inference_mode():
        # Calculate cosine similarity
        similarity_scores = F.cosine_similarity(target_movie_embedding, all_movie_embeddings, dim=1)

        # Sort movies based on similarity scores
        sorted_indices = torch.argsort(similarity_scores, descending=True)

        return sorted_indices[:top_n]

def more_movies(idx, n:int=10):
    if isinstance(idx, int):
        new_movie_id = dataset.movieId2idx[idx]
        print(f'Movie : {get_movie_name(idx, new_movie_df)}')

    elif isinstance(idx, str):
        new_movie_id = dataset.movieId2idx[get_movie_id(idx, new_movie_df)]
        print(f'Movie: {idx}')


    target_movie_embedding = model.movie_embedding(torch.tensor(new_movie_id).to(device)).unsqueeze(0)
    all_movie_embeddings = model.movie_embedding.weight.data

    # Find similar movies
    similar_movie_indices = find_similar_movies(target_movie_embedding, all_movie_embeddings, top_n=n+1)

    return [
        movie_df[movie_df.movieId == dataset.idx2movieId[i.item()]][
            'title'
        ].values[0]
        for num, i in enumerate(similar_movie_indices, 1)
    ]      

random_movie_id = int(new_user_df.sample(1).movieId.values[0])

random_movie_name = new_movie_df.sample(1).title.values[0]

# use this as search bar
search = 'avenger'
new_movie_df[new_movie_df.title.str.lower().str.contains(str(search))]

pp = ['Avengers, The (1998)','Avengers, The (2012)','Captain America: The First Avenger (2011)' ]

for i in pp:
    display(more_movies(idx = i))

# Our prediction looks perfect. 
# > Note: more the ratings well adjusted the embeddings, movies with less ratings can be seen randomly at any place, because there place in the embedding space is not adjusted enough as highly rated movies.




