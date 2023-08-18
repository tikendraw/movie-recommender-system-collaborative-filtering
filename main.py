import torch
import yaml
import pandas as pd
from src.data_loader import load_datasets
from src.model import RecommenderModel
from src.inference import more_movies

def main():
    # Load configurations
    with open('config/data_config.yaml', 'r') as data_config_file:
        data_config = yaml.safe_load(data_config_file)

    train_dataset, movie_rating_data = load_datasets(data_config)
    num_users = train_dataset.num_users
    num_movies = train_dataset.num_movies
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    new_movie_df = pd.read_csv(data_config['new_movie_df_path'])
    
    # Load model and configurations
    with open('config/hyperparameters.yaml', 'r') as config_file:
        hyperparameters = yaml.safe_load(config_file)

    embedding_dim = hyperparameters['embedding_dim']

    model = RecommenderModel(num_users, num_movies, embedding_dim)
    model_path = 'models/model.pth'  # Set the correct model path
    model.load_model(model_path)


    # Example of getting more movie recommendations
    search_movie = 'venger'  # Example movie search query
    search_results = new_movie_df[
        new_movie_df.title.str.lower().str.contains(search_movie)
    ]

    for idx, row in search_results.iterrows():
        similar_movies = more_movies(row['movieId'], movie_rating_data.movieId2idx, train_dataset.idx2movieId, new_movie_df, model, device=device, n = 10)
        print(f"For movie '{row['title']}' (ID: {row['movieId']}):")
        for i, movie_title in enumerate(similar_movies, start=1):
            print(f"{i}. {movie_title}")
        print('-' * 40)

if __name__ == "__main__":
    main()
