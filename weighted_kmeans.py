# -*- coding: utf-8 -*-
"""
Weighted K-Means
More see:
http://sanghv.com/download/Ebook/Machine%20Learning/FS/[Huan_Liu,_Hiroshi_Motoda]_Computational_methods_o(BookZZ.org).pdf
"""

import numpy as np

from k_init import _k_init
from sklearn.utils import check_random_state
from sklearn.utils.extmath import squared_norm


def _weightmatrix(k, m):
    """
    `matrix`: array, [k, m]
    """
    matrix = np.random.rand(k, m)
    matrix = matrix / matrix.sum(axis=1)[:,None]
    return matrix


def _tolerance(X, tol):
    """
    `X`: [n_samples, m_features]
    Return a tolerance which is independent of the data set
    """
    variances = np.var(X, axis=0)
    return np.mean(variances) * tol


class WeightedKMeans():

    def __init__(self, n_clusters=4, 
                 m=3, init_weights_method='random',
                 init_center_method='kmeans++', 
                 max_iter=300, 
                 random_state=0, tol=1e-6):
        """
        Parameters
        ----------
        n_clusters: int
            The number of clusters
            
        m: float
            weight parameter, m >=1 The closer to m is to 1, the closter to hard kmeans.
            
        max_iter: int
            Maximum number of iterations of the k-means algorithm for a
            single run.
        
        init_weights_method: 'fixed' or 'random'
        
        init_center_method: 'random', 'kmeans++'
          
        tot: float
            Relative tolerance with regards to inertia to declare convergence
            
        random_state : integer or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.
            
        Attributes
        ----------
        cluster_centers_ : array, [n_clusters, n_features]
            Coordinates of cluster centers

        labels_ :
            Labels of each point

        weights_ : array, [n_clusters, n_features]
            Weight of each cluster.
        
        """
        
        self.n_clusters = n_clusters
        assert m >= 1
        self.m = m
        self.init_weights_method = init_weights_method
        self.init_center_method = init_center_method
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol      
        
    def _update_label(self, X):
        """
        Fix Weights and Center, Update Label
        """
        n_samples, m_features = X.shape
        
        # distance between arbitrary point and all cluter center
        affiliation = np.zeros((n_samples, self.n_clusters))
        
        # calculate distance between data and center i
        for center_id, center in enumerate(self.cluster_centers_):
            center_i_dist = np.dot((X - center) ** 2, (self.weights_[center_id] ** self.m).T)
            # print(center_i_dist)
            affiliation[:, center_id] = center_i_dist.T
        
        # Set min distance () be the arbitray cluster label 
        self.labels_ = np.argmin(affiliation, axis=1)
    
    def _update_center(self, X):
        """
        Fix Weights and Labels, Update Centers
        """
        centers_old = self.cluster_centers_.copy()
        
        # choose data belong to cluster k
        # and update cluster center with it mean
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            self.cluster_centers_[k] = np.mean(X[mask], axis=0)
            
        center_shift_total = squared_norm(self.cluster_centers_ - centers_old) 
        return center_shift_total
    
    def _update_weight(self, X):
        """
        Fix Labels and Centers, Update Weights
        """
        # In case of D[i]=0, so add constant sigma
        sigma = np.mean(np.var(X, 0))
        n_samples, m_features = X.shape
        
        # D matrix 
        D = np.zeros((self.n_clusters, m_features))
        for k in range(self.n_clusters):
            mask = (self.labels_ == k) 
            # In case of D[i]==0, so add sigma
            D[k] = np.sum((X[mask] - self.cluster_centers_[k]) ** 2 + sigma, axis=0) 
            # D[k] = np.sum((X[mask] - self.cluster_centers_[k]) ** 2, axis=0) 
        for k in range(self.n_clusters):
            for j in range(m_features):
                self.weights_[k][j] = 1 / np.sum((D[k][j] / D[k]) ** (1 / (self.m - 1)))
               
    def fit(self, X, y=None):
        n_samples, m_features = X.shape
        
        # Variance
        variances = np.mean(np.var(X, 0))
        self.tol *= variances 
        
        # Initialize weight matrix
        if self.init_weights_method == 'random':
            self.weights_ = _weightmatrix(self.n_clusters, m_features)
        elif self.init_weights_method == 'fixed':
            self.weights_ = np.ones((self.n_clusters, m_features)) * (1 / m_features)
        else:
            raise Exception('init_weights_method must be `random` or `fixed`')
        
        print('origin weights_ \n', self.weights_)
        
        # Initialize center 
        if self.init_center_method == 'kmeans++':
            random_state = check_random_state(self.random_state)
            self.cluster_centers_ = _k_init(X=X, 
                                            n_clusters=self.n_clusters, 
                                            random_state=random_state)         
        elif self.init_center_method == 'random':
            random_state = check_random_state(self.random_state)
            chosen_ids = random_state.permutation(n_samples)[:self.n_clusters]
            self.cluster_centers_ = X[chosen_ids]
        else:
            raise Exception('init_center_method must be `random` or `kmeans++`')
        
        print('origin center_ \n', self.cluster_centers_)
        
        # Iteration
        for i in range(self.max_iter):
            self._update_label(X)
            center_shift_total = self._update_center(X)
            print('Iterion %i cluster_centers_ \n' % i, self.cluster_centers_)
            print('Iterion %i tolerance: ' % i, center_shift_total)
            if center_shift_total < self.tol:
                break
            self._update_weight(X) 
            print('Iterion %i weights_\n' % i, self.weights_)
            print('\n')
        else:
            print('Iterion Completed')
        return self


