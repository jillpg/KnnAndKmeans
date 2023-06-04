__authors__ = ['1604284','1598906']
__group__ = 'GrupDL.17'

import collections

import numpy as np
import scipy.spatial.distance

import utils

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################






    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################



        if float != X.dtype :
            X=X.astype(float)
        if X is list:
            X=X.reshape(len(X),3)
        elif X.shape[2] == 3:
            X= X.reshape(X.shape[0]*X.shape[1],3)

        self.X = X



    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = 1000
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################





    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.centroids=np.zeros((self.K, 3))
        self.old_centroids=np.zeros((self.K, 3))
        if self.options["km_init"]=="first": # pillar las K primeras filas que no se repiten
            self.centroids=self.X[np.sort((np.unique(self.X,axis=0,return_index=True))[1])[:self.K]]

        elif self.options["km_init"] == "random": #pillar centroides randoms que no se repiten

            pixelsDiferentes=np.unique(self.X, axis=0)

            self.centroids = pixelsDiferentes[np.random.choice(range(pixelsDiferentes.shape[0]), self.K, False)]

        elif self.options["km_init"] == "custom": #pillar colores que mas se repiten

            
            pixelsDiferentes,counter=np.unique(self.X, axis=0,return_counts=True)

            mas_repetidos= pixelsDiferentes[np.flip(np.argsort(counter))[:self.K]]

            self.centroids = mas_repetidos








    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.labels=np.argmin(distance(self.X,self.centroids),axis=1)



    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids=np.copy(self.centroids)
        for k in np.arange(self.K):
            claseK = self.X[np.where(self.labels == k)]
            self.centroids[k]= np.mean(claseK,axis=0)





    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        if np.allclose(self.centroids,self.old_centroids,rtol=self.options["tolerance"]):
            return True

        return False


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self._init_centroids()
        for itera in np.arange(self.options["max_iter"]):
            self.get_labels()
            self.get_centroids()
            self.num_iter+=1
            if self.converges():
                break


    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """

        if self.options['fitting']=='WCD':
            tot = 1 / len(self.X)
            cVal = np.sum(np.square(np.amin(distance(self.X, self.centroids), axis=1)))
            rVal = np.multiply(tot, cVal)
        elif self.options['fitting']=='inter-class distance':
            tot = 1 / len(self.X)
            cVal = np.sum(np.square(np.amax(distance(self.X, self.centroids), axis=1)))
            rVal = np.multiply(tot, cVal)

        elif self.options['fitting']=='inter/intra-class variance':
            cVal_intra = np.sum(np.square(np.amin(distance(self.X, self.centroids), axis=1)))
            intra = np.multiply((1 / len(self.X)), cVal_intra)

            cVal_inter = np.sum(np.square(np.amax(distance(self.X, self.centroids), axis=1)))
            inter = np.multiply((1 / len(self.X)), cVal_inter)

            rVal=inter/intra





        return rVal




        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        WCD = np.zeros(max_K+1)
        DEC = np.zeros(max_K+1)
        for k in np.arange(2,max_K+1):
            self.K=k
            self.fit()
            WCD[k]=self.whitinClassDistance()
            if k>2:
                DEC[k] = 100 * (WCD[k] / WCD[k - 1])
        self.K = max_K

        for k in np.arange(2, max_K):
            if 100-DEC[k+1] < 27:
                self.K=k
                break

def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    dist=np.zeros((X.shape[0], C.shape[0]))

    dist=scipy.spatial.distance.cdist(X, C, 'euclidean')
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    labels= np.argmax(utils.get_color_prob(centroids),axis=1)



    return utils.colors[labels]
