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

'''plot the values of the columns'''
def plotVals(df):
    x = df['X'].values
    y = df['Y'].values

    '''create a plot and two subplots for -1 and 1'''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)

    '''iterate over the x,y values and tag and plot them appropriately'''
    for index in range(len(x)):
        if y[index] == 0:
            ax1.scatter(x[index], y[index], alpha=0.7, c='red', edgecolors='none', s=50, label='red')
        else:
            ax2.scatter(x[index], y[index], alpha=0.7, c='black', edgecolors='none', s=50, label='black')

    plt.title("Logistic Regression: Probability of Failure")
    plt.xlabel("Weeks of laziness")
    plt.ylabel("Pass / Fail")
    plt.show()


'''This function will calculate the gradient average'''
def computeGradientAverage(df, weights):
    x = df['X'].values
    y = df['Y'].values
    N = len(x)
    summation = 0

    for i in range(len(x)):
        top = y[i] * x[i] # Yn * Xn
        wTx = weights[0] + weights[1] * x[i] # ****NOTE: still slightly unsure about this
        y_wTx = y[i] * wTx # Yn * wTx
        bottom = 1 + math.pow(math.e, y_wTx) # 1 + e^y_n * wTx
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

    while (i < numIterations):
        g_t = computeGradientAverage(df, weights)
        v_t = g_t * -1
        weights = updateWeights(weights, learningRate, v_t)



'''this is the decision boundary'''
def decisionBoundary(probability):
    return 1 if probability >= .5 else 0

'''Just to have fun and make predictions once the weights are trained'''
def probabilityOutputFunction(trainedWeights, xNum):
    prediction = trainedWeights[0] + trainedWeights[1]*xNum
    prediction *= -1
    return 1 / 1 + math.pow(math.e, prediction)


def main():
    weights = [0, 0.05]
    df = createTrainingData()
    plotVals(df)


main()