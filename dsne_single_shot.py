#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:59:22 2017

@author: dsaha
"""
import numpy as np
import json
import argparse


def Hbeta(D=np.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the
    precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to
    no_dims dimensions."""

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to no_dims dimensions. The syntaxis of the function is
    Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 7
    # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[list(range(n)), list(range(n))] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(
                np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y),
                0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * (
            (dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (iter + 1), ": error is ", C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4  # exaggerated by 7 but divided by 4 ??

    # Return solution
    return Y


def tsne1(Shared_length,
          X=np.array([]),
          Y=np.array([]),
          no_dims=2,
          initial_dims=50,
          perplexity=30.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to no_dims dimensions. The syntaxis of the function is
    Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    def updateS(Y, G):
        return Y

    def updateL(Y, G):
        return Y + G

    def demeanS(Y):
        return Y

    def demeanL(Y):
        return Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)  # ask about this: why not use the Y passed
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 7
    # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[list(range(n)), list(range(n))] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(
                np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y),
                0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * (
                (dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)

        Y[:Shared_length, :] = updateS(Y[:Shared_length, :],
                                       iY[:Shared_length, :])
        Y[Shared_length:, :] = updateL(Y[Shared_length:, :],
                                       iY[Shared_length:, :])
        Y[:Shared_length, :] = demeanS(Y[:Shared_length, :])
        Y[Shared_length:, :] = demeanL(Y[Shared_length:, :])

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (iter + 1), ": error is ", C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4  # exaggerated by 7 but divided by 4 ??

    return Y  # Return solution


def normalize_columns(arr=np.array([])):
    rows, cols = arr.shape
    for row in range(rows):
        p = abs(arr[row, :]).max()
        arr[row, :] = (arr[row, :] / p) if p else arr[row, :]

    return arr


def local_site(args, computation_phase):

    shared_X = np.loadtxt(args["shared_X"])
    shared_Y = np.loadtxt(args["shared_Y"])
    no_dims = args["no_dims"]
    initial_dims = args["initial_dims"]
    perplexity = args["perplexity"]
    sharedRows, sharedColumns = shared_X.shape

    # load high dimensional site 1 data
    parser = argparse.ArgumentParser(
        description='''read in coinstac args for local computation''')
    parser.add_argument('--run', type=json.loads, help='grab coinstac args')
    localSite1_Data = ''' {
        "site1_Data": "Site_1_Mnist_X.txt",
        "site1_Label": "Shared_label.txt"
    } '''
    site1args = parser.parse_args(['--run', localSite1_Data])
    Site1Data = np.loadtxt(site1args.run["site1_Data"])
    (site1Rows, site1Columns) = Site1Data.shape

    # load label of site 1 data
    #    site1Label = np.loadtxt(site1args.run["site1_Label"])

    # create combinded list by local and remote data
    combined_X = np.concatenate((shared_X, Site1Data), axis=0)
    combined_X = normalize_columns(combined_X)

    # create low dimensional position
    combined_Y = np.random.randn(combined_X.shape[0], no_dims)
    combined_Y[:shared_Y.shape[0], :] = shared_Y

    Y_plot = tsne1(
        sharedRows,
        combined_X,
        combined_Y,
        no_dims=no_dims,
        initial_dims=initial_dims,
        perplexity=perplexity)

    # save local site data into file
    f1 = open("local_site1.txt", "w")
    for i in range(sharedRows, len(Y_plot)):
        f1.write(str(Y_plot[i][0]) + '\t')
        f1.write(str(Y_plot[i][1]) + '\n')
    f1.close()

    # pass data to remote in json format
    localJson = ''' {"local": "local_site1.txt"} '''
    localY = parser.parse_args(['--run', localJson])

    return (localY.run)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='''read in coinstac args for remote computation''')
    parser.add_argument('--run', type=json.loads, help='grab coinstac args')

    # load high dimensional shared data
    sharedData = ''' {a
        "shared_X": "Shared_Mnist_X.txt",
        "shared_Label": "Shared_Label.txt",
        "no_dims": 2,
        "initial_dims": 50,
        "perplexity" : 20.0
    } '''
    args = parser.parse_args(['--run', sharedData])

    shared_X = np.loadtxt(args.run["shared_X"])
    sharedLabel = np.loadtxt(args.run["shared_Label"])
    no_dims = args.run["no_dims"]
    initial_dims = args.run["initial_dims"]
    perplexity = args.run["perplexity"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    # shared data computation in tsne
    shared_Y = tsne(shared_X, no_dims, initial_dims, perplexity)

    with open("Y_values.txt", "w") as f:
        for i in range(0, len(shared_Y)):
            f.write(str(shared_Y[i][0]) + '\t')
            f.write(str(shared_Y[i][1]) + '\n')

    args.run["shared_Y"] = "Y_values.txt"

    # receive data from local site
    L1 = local_site(args.run, computation_phase=0)
    LY = np.loadtxt(L1["local"])
