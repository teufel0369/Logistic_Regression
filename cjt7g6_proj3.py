import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import math

'''create the training data set'''
def createTrainingData():
    x = [1, 2, 3, 4, 5, 6, 7, 8] #create the x-values
    y = [0, 1, 0, 1, 0, 1, 1, 1] #create the y-values

    df = pd.DataFrame() #create an empty dataframe to append the lists to
    df['X'] = x #insert the list of x-values into the dataframe
    df['Y'] = y #insert the list of y-values into the dataframe

    return df

'''This function will calculate the gradient average'''
def computeGradientAverage(df, weights):
    x = df['X'].values
    y = df['Y'].values
    N = len(x)
    summation = 0

    for i in range(len(x)):
        top = y[i] * x[i] # Yn * Xn
        wTx = weights[0]*1 + weights[1] * x[i] # compute the dot product of
        y_wTx = y[i] * wTx # Yn * wTx
        bottom = 1 + math.exp(y_wTx) # 1 + e^y_n * wTx
        result = top / bottom #divide the result
        summation += result #add it to the sum

    g_t = summation / N #compute the average
    g_t *= -1 #multiply it by -1 per the algorithm on page 95

    return g_t

'''this function will update the weights'''
def updateWeights(weights, learningRate, vT):

    # w(t+1) = w(t) + learningRate*vT where vT = the gradient direction
    weights[0] = weights[0] + learningRate*vT
    weights[1] = weights[1] + learningRate*vT

    return weights

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
def probabilityOutputFunction(trainedWeights, xNum):
    prediction = trainedWeights[0] + trainedWeights[1]*xNum
    prediction *= -1
    return 1 / 1 + math.exp(prediction)



def main():
    weights = [0, 0.05]
    df = createTrainingData()
    weights = trainWeights(df, weights, numIterations=1000, learningRate=0.01)
    string = "\nSigmoid: " + str(1) + " / " + str(1) + " + e ^ " + str(weights[0]) + " + " + str(weights[1]) + "x"
    print(string)

main()