import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import math


def createTrainingData():
    x = [1, 2, 3, 4, 5, 6, 7, 8] #create the x-values
    y = [0, 1, 0, 1, 0, 1, 1, 1] #create the y-values

    df = pd.DataFrame() #create an empty dataframe to append the lists to
    df['X'] = x #insert the list of x-values into the dataframe
    df['Y'] = y #insert the list of y-values into the dataframe

    return df

#plot the values of the columns
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

#Create a hyperbolic tangent soft-threshold
def sigmoid(score):
    negScore = score * -1
    term1 = math.pow(math.e, score)
    term2 = math.pow(math.e, negScore)

    top = term1 - term2
    bottom = term1 + term2

    theta = top / bottom

    return theta

#this function will update the weights
def updateWeights(weights, learningRate, vT):

    # w(t+1_ = w(t) + learningRate*vT where vT = the gradient direction
    weights[0] = weights[0] + learningRate*vT
    weights[1] = weights[1] + learningRate*vT

    return weights

#this function will compute the gradient that we need to travel
def computeGradient(df):
    



def main():
    df = createTrainingData()
    plotVals(df)


main()