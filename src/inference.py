import torch
import torch.nn.functional as F
from data_loader import MovieRatingData


def find_similar_movies(target_movie_embedding, all_movie_embeddings, top_n=5):
    with torch.inference_mode():
        # Calculate cosine similarity
        similarity_scores = F.cosine_similarity(
            target_movie_embedding, all_movie_embeddings, dim=1
        )

        # Sort movies based on similarity scores
        sorted_indices = torch.argsort(similarity_scores, descending=True)

        return sorted_indices[:top_n]


def more_movies(idx, movie_rating_info: MovieRatingData, model, device, n=10):
    if isinstance(idx, int):
        new_movie_id = movie_rating_info.movieId2idx[idx]
        print(f"Movie: {get_movie_name(idx,movie_rating_info.movie_df)}")

    elif isinstance(idx, str):
        new_movie_id = int(get_movie_id(idx, movie_rating_info.movie_df))
        new_movie_id = movie_rating_info.movieId2idx[new_movie_id]
        print(f"Movie: {idx}")

    model.to(device)
    target_movie_embedding = model.movie_embedding(
        torch.tensor(new_movie_id).to(device)
    ).unsqueeze(0)
    all_movie_embeddings = model.movie_embedding.weight.data

    # Find similar movies
    similar_movie_indices = find_similar_movies(
        target_movie_embedding, all_movie_embeddings, top_n=n + 1
    )

    return [
        movie_rating_info.movie_df[
            movie_rating_info.movie_df.movieId
            == movie_rating_info.idx2movieId[i.item()]
        ]["title"].values[0]
        for num, i in enumerate(similar_movie_indices, 1)
    ]


def get_movie_name(idx, df):
    try:
        return df[df.movieId == idx].title.values[0]
    except IndexError as e:
        print("IndexError:", idx)


def get_movie_id(movie_name, df):
    try:
        return df[df.title == movie_name].movieId.values[0]
    except IndexError as e:
        print("Movie not found in dataset")
