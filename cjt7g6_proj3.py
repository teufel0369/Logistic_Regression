import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import math
import decimal

'''create the training data set'''
def createTrainingData():
    x = [1, 2, 3, 4, 5, 6, 7, 8] #create the x-values
    y = [0, 1, 0, 1, 0, 1, 1, 1] #create the y-values

    df = pd.DataFrame() #create an empty dataframe to append the lists to
    df['X'] = x #insert the list of x-values into the dataframe
    df['Y'] = y #insert the list of y-values into the dataframe

    return df

'''This function will create the training data for question c'''
def createTrainingData2():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 3, 5]  # create the x-values
    y = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1]  # create the y-values

    df = pd.DataFrame()  # create an empty dataframe to append the lists to
    df['X'] = x  # insert the list of x-values into the dataframe
    df['Y'] = y  # insert the list of y-values into the dataframe

    return df


'''This function will calculate wTx'''
def wTxEquation(df, weights):
    x = df['X'].values
    wTx = weights[0] + weights[1]*1

    for i in range(len(x)):
        wTx += weights[1] * x[i]

    return wTx

'''This function will calculate the gradient average'''
def computeGradientAverage(df, weights):
    x = df['X'].values
    y = df['Y'].values
    N = len(x)
    summation = 0
    wTx = wTxEquation(df, weights)

    for i in range(len(x)):
        top = y[i] * x[i] # Yn * Xn
        y_wTx = y[i] * wTx # Yn * wTx
        bottom = 1 + (math.exp(y_wTx)) # 1 + e^y_n * wTx
        result = top / bottom #divide the result
        summation += result #add it to the sum

    g_t = (-1*summation) / N #compute the average

    return g_t

'''this function will update the weights'''
def updateWeights(weights, learningRate, vT):

    # w(t+1) = w(t) + learningRate*vT where vT = the gradient direction
    weights[0] = weights[0] + learningRate*vT
    weights[1] = weights[1] + learningRate*vT

    return weights

'''this function will train the weights'''
def trainWeights(df, weights, numIterations, learningRate):
    i = 1

    '''This loop will continue to train the weights until it maxes out the number of iterations'''
    while (i < numIterations):
        g_t = computeGradientAverage(df, weights) #compute the gradient
        v_t = g_t * -1 #set the direction to move
        weights = updateWeights(weights, learningRate, v_t) #update the weights
        print("Weights for iteration: " + str(i))
        print(weights)
        print("")
        i += 1 #increment the counter

    return weights

'''this is the decision boundary'''
def decisionBoundary(probability):
    return 1 if probability >= .5 else 0

'''Just to make predictions once the weights are trained'''
def sigmoid(trainedWeights, xNum):
    prediction = -1 * (trainedWeights[0] + trainedWeights[1]*xNum)
    top = 1
    bottom = 1 + math.exp(prediction)
    return top / bottom


def main():
    weights = [random.uniform(0.001, 0.1), random.uniform(0.001, 0.1)] #initialize the weights to some random small values
    df = createTrainingData() #create the training data
    weights = trainWeights(df, weights, numIterations=1000, learningRate=0.05) #train the weights
    string = "\nSigmoid: " + str(1) + " / " + str(1) + " + e ^ -(" + str(weights[0]) + " + " + str(weights[1]) + "x)"
    print(string) #print out the equation of a line
    threeWeeksNoStudy = sigmoid(weights, 3) * 100
    fiveWeeksNoStudy = sigmoid(weights, 5) * 100
    threeWeekString = "\nThe probability of passing with 3 weeks of no study: " + str(threeWeeksNoStudy) + "%"
    fiveWeekString = "\nThe probability of passing with 5 weeks of no study: " + str(fiveWeeksNoStudy) + "%"
    print(threeWeekString)
    print(fiveWeekString)

    weights2 = [random.uniform(0.001, 0.1), random.uniform(0.001, 0.1)]  # initialize the weights to some random small values
    df2 = createTrainingData()  # create the training data
    weights2 = trainWeights(df2, weights2, numIterations=1000, learningRate=0.05)  # train the weights
    string = "\nSigmoid: " + str(1) + " / " + str(1) + " + e ^ -(" + str(weights2[0]) + " + " + str(weights2[1]) + "x)"
    print(string)  # print out the equation of a line
    threeWeeksNoStudy = sigmoid(weights2, 3) * 100
    fiveWeeksNoStudy = sigmoid(weights2, 5) * 100
    threeWeekString = "\nThe probability of passing with 3 weeks of no study: " + str(threeWeeksNoStudy) + "%"
    fiveWeekString = "\nThe probability of passing with 5 weeks of no study: " + str(fiveWeeksNoStudy) + "%"
    print(threeWeekString)
    print(fiveWeekString)

main()