
import collections
import statistics

import numpy as np
import math
import operator

from utils import rgb2gray
import pandas.core.algorithms
import scipy.spatial.distance
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
     


    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        self.train_data = np.array(train_data.reshape((train_data.shape[0],14400)),dtype=float)


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        self.neighbors = self.labels[np.argsort(scipy.spatial.distance.cdist(np.array(test_data.reshape((test_data.shape[0],14400)),dtype=float),self.train_data,metric='euclidean'),axis=1)[:,:k]]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """

        clase=np.empty(10,dtype=object)
        porcentaje=np.empty(10,dtype=float)

        sorted=np.sort(self.neighbors,axis=1)

        for row in range(self.neighbors.shape[0]):
            clase[row]=statistics.mode(sorted[row])
            contador=collections.Counter(sorted[row])
            porcentaje[row]=contador[clase[row]]/sorted[row].shape[0]

        return clase #,porcentaje  #quitar el si es necesario


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """


        self.get_k_neighbours(test_data,k)

        return self.get_class()
