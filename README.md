# Movie Recommender System
==============================

A movie recommender system based on item-collaborative-filtering implemented using PyTorch. This project demonstrates how to build a recommendation model using collaborative filtering techniques and embeddings to provide personalized movie recommendations based on user ratings.

## Table of Contents

- [Concept](#concept)
- [Usefulness](#usefulness)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Concept

The movie recommender system uses collaborative filtering to recommend movies to users based on their past movie ratings and the preferences of similar users. It employs embeddings to represent users and movies in a lower-dimensional space, where the dot product between user and movie embeddings gives the predicted rating.

## Usefulness

- **Personalized Recommendations:** The system provides personalized movie recommendations to users based on their preferences and behavior.

- **Discovery:** Users can discover new movies they might enjoy but haven't watched before.

- **Enhanced User Experience:** By suggesting relevant movies, the system enhances the user experience and encourages more engagement with the platform.


## Project Organization
------------

├── config
│   ├── data_config.yaml
│   └── hyperparameters.yaml
├── data
│   ├── link.csv
│   ├── movie.csv
│   ├── movies.csv
│   ├── new_movie_df.csv
│   ├── new_user_df.csv
│   ├── plot_embedding.csv
│   ├── rating.csv
│   └── tag.csv
├── LICENSE
├── main.py
├── models
│   └── model.pth
├── notebooks
│   ├── main.py
│   ├── recommeder_nb.ipynb
│   └── recommeder_nb_main.ipynb
├── pre-commit.sh
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── data_loader.py
│   ├── inference.py
│   ├── __init__.py
│   └── model.py
├── tox.ini
└── train.py




--------

## Getting Started

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/tikendraw/movie-recommender-system-collaborative-filtering.git
   cd movie-recommender-system-collaborative-filtering

2. Install the required dependencies:
```
pip install -r requirements.txt
```
### Usage
* Update the data paths in the config/data_config.yaml file.

* Configure hyperparameters in the config/hyperparameters.yaml file.

* Train the model using:
```
python train.py
```
* Run the recommender system:

```
python main.py
```

## Contribution
Contributions are welcome! Feel free to open issues and submit pull requests for improvements.

currently on working on different deep learning projects. But i intend to make an actual movie recommending website and implement all sorts of recommendation, feel free to approach and contribute to the project.

