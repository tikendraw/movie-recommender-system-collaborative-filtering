import torch
from torch import nn
from pathlib import Path

class RecommenderModel(nn.Module):
    def __init__(self, num_users:int, num_movies:int, embedding_dim:int, model_path:Path=None):
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
        output = self.out(interaction) 
        if debug:
            print('user_emb.shape   : ',user_emb.shape)
            print('movie_emb.shape  : ',movie_emb.shape)
            print('interaction.shape: ',interaction.shape)
            
            print('user_emb_bias.shape  : ',user_emb_bias.shape)
            print('movie_emb_bias.shape: ',movie_emb_bias.shape)
            
            print('output.shape     :',output.shape)

        return output
    
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