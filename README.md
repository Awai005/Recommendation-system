# Recommendation System

This repository contains Jupiter Notebook code for building a recommendation system using collaborative filtering techniques. The recommendation system includes user-user based collaborative filtering, item-item based collaborative filtering, and model-based collaborative filtering.

## Features
- **User-User Collaborative Filtering**
- **Item-Item Collaborative Filtering**
- **Model-Based Collaborative Filtering**

## Usage
Running the Jupyter Notebook
Ensure you have Jupyter installed. If not, install it using pip:

```bash
pip install notebook
```
Launch the Jupyter Notebook:
```bash
jupyter notebook
```

Open the recommendation_system.ipynb notebook and run the cells to see the implementation and results of the recommendation system.

## Code Overview
### Data Preparation
The code reads the dataset from an Excel file and prepares the user-item matrix

### User-User Collaborative Filtering
Calculate user similarity and make predictions:
```bash
user_similarity = pairwise_distances(train_matrix, metric='cosine')

def predict_user_user(train_matrix, user_similarity, n_similar=100):
    similar_n = user_similarity.argsort()[:,-n_similar:][:,::-1]
    pred = np.zeros((n_users, n_questions))
    for i, users in enumerate(similar_n):
        similar_users_indexes = users
        similarity_n = user_similarity[i, similar_users_indexes]
        matrix_n = train_matrix[similar_users_indexes, :]
        rated_items = similarity_n[:, np.newaxis].T.dot(matrix_n - matrix_n.mean(axis=1)[:, np.newaxis]) / similarity_n.sum()
        pred[i, :] = rated_items
    return pred

predictions = predict_user_user(train_matrix, user_similarity, 100) + train_matrix.mean(axis=1)[:, np.newaxis]
```
### Item-Item Collaborative Filtering
Calculate item similarity and make predictions:

``` bash
item_similarity = pairwise_distances(train_matrix.T, metric='cosine')

def predict_item_item(train_matrix, item_similarity, n_similar=100):
    similar_n = item_similarity.argsort()[:,-n_similar:][:,::-1]
    pred = np.zeros((n_users, n_questions))
    for i, items in enumerate(similar_n):
        similar_items_indexes = items
        similarity_n = item_similarity[i, similar_items_indexes]
        matrix_n = train_matrix[:, similar_items_indexes]
        rated_items = matrix_n.dot(similarity_n) / similarity_n.sum()
        pred[:, i] = rated_items
    return pred

predictions = predict_item_item(train_matrix, item_similarity, 100)

```

## Motivation
This project was motivated by interest in machine learning.

