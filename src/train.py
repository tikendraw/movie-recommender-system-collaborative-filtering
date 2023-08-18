import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time
import yaml

from model import RecommenderModel
from data_loader import load_datasets


def train_model(
    model,
    dataloader,
    optimizer,
    loss_function,
    num_epochs=10,
    device="cpu",
    data_percent=1.0,
    steps_per_epoch=None,
):
    model.to(device)
    print(f"{model.__class__.__name__} Running on: {device}")

    data_size = int(data_percent * len(dataloader))
    dataloader = iter(dataloader)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        epoch_progress = tqdm(
            range(data_size), desc=f"Epoch [{epoch+1:2}/{num_epochs:2}]"
        )

        if steps_per_epoch is not None:
            epoch_progress = tqdm(
                range(steps_per_epoch), desc=f"Epoch [{epoch+1:2}/{num_epochs:2}]"
            )

        last_update_time = time.time() - 1.0  # Initialize to ensure the first update

        for _ in epoch_progress:
            try:
                batch = next(dataloader)
            except StopIteration:
                print("Dataloader is exhausted. Resetting or stopping training.")
                # You might want to break the loop or take some other action here
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
                epoch_progress.set_postfix(
                    {"Loss": formatted_loss, "MSE": formatted_mse, "MAE": formatted_mae}
                )
                epoch_progress.update()
                last_update_time = current_time

            if steps_per_epoch is not None and _ + 1 >= steps_per_epoch:
                break

        # epoch_progress.close()
        average_loss = (
            total_loss / min(data_size, steps_per_epoch)
            if steps_per_epoch is not None
            else total_loss / data_size
        )
        average_mse = (
            total_mse / min(data_size, steps_per_epoch)
            if steps_per_epoch is not None
            else total_mse / data_size
        )
        average_mae = (
            total_mae / min(data_size, steps_per_epoch)
            if steps_per_epoch is not None
            else total_mae / data_size
        )

        print(
            f"Epoch [{epoch+1:2}/{num_epochs:2}] - Average Loss: {average_loss:.8f} - Average MSE: {average_mse:.8f} - Average MAE: {average_mae:.8f}"
        )
        print()


def train():
    # Load configurations
    with open("config/hyperparameters.yaml", "r") as config_file:
        hyperparameters = yaml.safe_load(config_file)

    embedding_dim = hyperparameters["embedding_dim"]
    learning_rate = hyperparameters["learning_rate"]
    batch_size = hyperparameters["batch_size"]
    num_epochs = hyperparameters["num_epochs"]
    data_percent = hyperparameters["data_percent"]
    # Load data and configurations
    with open("config/data_config.yaml", "r") as data_config_file:
        data_config = yaml.safe_load(data_config_file)

    train_dataset, _ = load_datasets(data_config)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_users = train_dataset.num_users
    num_movies = train_dataset.num_movies
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize your model, optimizer, and loss function
    model = RecommenderModel(num_users, num_movies, embedding_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    # Train the model
    train_model(
        model, dataloader, optimizer, loss_function, num_epochs, device, data_percent
    )


if __name__ == "__main__":
    train()
