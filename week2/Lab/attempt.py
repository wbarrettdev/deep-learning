   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gradient_descent(bias, lambda1, alpha, X, Y, max_iter):
    pass


def linearRegression(X, Y):
    
    # set initial parameters for model
    bias = 0
    lambda1 = 0
    
    alpha = 0.005 # learning rate
    max_iter=50

    gradient_descent(bias, lambda1, alpha, X, Y, max_iter)
    #TODO
    # call gredient decent to calculate intercept(=bias) and slope(lambda1)
    #bias, lambda1 = gradient_descent(bias, lambda1, alpha, X, Y, max_iter)
    #print ('Final bias and  lambda1 values are = ', bias, lambda1, " respecively." )
    
    # plot the data and overlay the linear regression model
    yPredictions = (lambda1*X)+bias
    plt.scatter(X, Y)
    plt.plot(X,yPredictions,'k-')
    plt.show()


def visualizeData(X, Y):
    plt.plot(X, Y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    
    
def main():

    # Read data into a dataframe
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data.xlsx')
    df = pd.read_excel(data_path)
    df = df.dropna() 

    # Store feature and target data in separate arrays
    Y = df['Y'].values
    X = df['X'].values
    
    visualizeData(X, Y)


    # Perform standarization on the feature data
    X = (X - np.mean(X))/np.std(X)
    
    linearRegression(X, Y)
    

if __name__ == "__main__":
    main()