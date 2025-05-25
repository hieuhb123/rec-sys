import os
import numpy as np
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from RBMAlgorithm import RBMAlgorithm

def recommend_for_user(algo, trainset, user_raw_id, hotel_map, n=5):
    # Get all hotel raw ids
    all_hotel_ids = set(hotel_map.keys())
    # Get hotels the user has already rated
    try:
        user_inner_id = trainset.to_inner_uid(str(user_raw_id))
        rated_inner_ids = set(j for (j, _) in trainset.ur[user_inner_id])
        rated_raw_ids = set(trainset.to_raw_iid(iid) for iid in rated_inner_ids)
    except ValueError:
        rated_raw_ids = set()
    # Recommend hotels not yet rated
    unrated_hotel_ids = all_hotel_ids - set(map(int, rated_raw_ids))
    predictions = []
    for hotel_id in unrated_hotel_ids:
        pred = algo.predict(str(user_raw_id), str(hotel_id))
        predictions.append((hotel_id, pred.est))
    # Sort by predicted rating
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    print(f"\nTop {n} recommendations for user {user_raw_id}:")
    for hotel_id, est_rating in top_n:
        print(f"HotelID: {hotel_id}, Hotel Name: {hotel_map[hotel_id]}, Predicted Rating: {est_rating:.2f}")

def test_rbmalgorithm_with_output_csv():
    # Build HotelID to Name mapping
    df = pd.read_csv("output.csv")
    hotel_map = dict(zip(df["HotelID"], df["Name Hotel"]))

    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_file("ratings.csv", reader=reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    algo = RBMAlgorithm(epochs=2, hiddenDim=10, learningRate=0.01, batchSize=10)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    mae = accuracy.mae(predictions, verbose=True)
    print(f"RBMAlgorithm RMSE: {rmse}")
    print(f"RBMAlgorithm MAE: {mae}")

    # Example: recommend for test user with UserID 1983
    recommend_for_user(algo, trainset, user_raw_id=1983, hotel_map=hotel_map, n=5)

if __name__ == "__main__":
    test_rbmalgorithm_with_output_csv()