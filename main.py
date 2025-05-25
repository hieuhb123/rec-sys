import os
import numpy as np
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from RBMAlgorithm import RBMAlgorithm
from RecommenderMetrics import RecommenderMetrics

from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

def Evaluated(algo, data, n=5):

    fullTrainSet = data.build_full_trainset()
    fullAntiTestSet = fullTrainSet.build_anti_testset()
    
    #Build a "leave one out" train/test split for evaluating top-N recommenders
    #And build an anti-test-set for building predictions
    LOOCV = LeaveOneOut(n_splits=1, random_state=1)
    for train, test in LOOCV.split(data):
        LOOCVTrain = train
        LOOCVTest = test
        
    LOOCVAntiTestSet = LOOCVTrain.build_anti_testset()
    sim_options = {'name': 'cosine', 'user_based': False}
    simsAlgo = KNNBaseline(sim_options=sim_options)
    simsAlgo.fit(fullTrainSet)

    metrics = {}
    trainset, testset = train_test_split(data, test_size=0.25, random_state=1)
    algo.fit(trainset)
    predictions = algo.test(testset)
    # rmse = accuracy.rmse(predictions, verbose=True)
    # mae = accuracy.mae(predictions, verbose=True)
    metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
    metrics["MAE"] = RecommenderMetrics.MAE(predictions)

    algo.fit(LOOCVTrain)
    leftOutPredictions = algo.test(LOOCVTest)        
    # Build predictions for all ratings not in the training set
    allPredictions = algo.test(LOOCVAntiTestSet)
    # Compute top 10 recs for each user
    topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
    # See how often we recommended a movie the user actually rated
    metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)   
    # See how often we recommended a movie the user actually liked
    metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
    # Compute ARHR
    metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)

    
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "RMSE", "MAE", "HR", "cHR", "ARHR"))
    print("{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"]))
    print("\nLegend:\n")
    print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
    print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
    print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
    print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
    print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.\n" )

def recommend_for_user(algo, data, hotel_map, testSubject=85, k=10):
    
    trainSet = data.build_full_trainset()
    algo.fit(trainSet)
    
    testSet = GetAntiTestSetForUser(data, testSubject)

    predictions = algo.test(testSet)
    
    recommendations = []
    
    print ("\nI recommend for user "+str(testSubject)+": ")
    for userID, hotelID, actualRating, estimatedRating, _ in predictions:
        intHotelID = int(hotelID)
        recommendations.append((intHotelID, estimatedRating))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    for ratings in recommendations[:k]:
        print(hotel_map[ratings[0]], ratings[1])

def GetAntiTestSetForUser(data, testSubject):
    trainset = data.build_full_trainset()
    fill = trainset.global_mean
    anti_testset = []
    u = trainset.to_inner_uid(str(testSubject))
    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                i in trainset.all_items() if
                                i not in user_items]
    return anti_testset

if __name__ == "__main__":

    testSubject = 1284  # Example user ID for testing

    df = pd.read_csv("output.csv")
    hotel_map = dict(zip(df["HotelID"], df["Name_Hotel"]))

    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_file("ratings.csv", reader=reader)
    
    algo = RBMAlgorithm(epochs=20, hiddenDim=10, learningRate=0.01, batchSize=10)

    # Evaluate the RBMAlgorithm
    Evaluated(algo, data)
    # Recommend hotels for the specified user
    recommend_for_user(algo, data, hotel_map, testSubject=testSubject, k=5)
