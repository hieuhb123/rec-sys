import os
import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from RBMAlgorithm import RBMAlgorithm

# test_RBMAlgorithm.py



def test_rbmalgorithm_with_output_csv():
    # Adjust delimiter and rating_scale if needed
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

if __name__ == "__main__":
    test_rbmalgorithm_with_output_csv()