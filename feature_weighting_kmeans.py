#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Weighted Kmeans
the weight is fixed!
More see 
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
    matrix = matrix / matrix.sum(axis=1)[:, None]
    return matrix

def kmeans_to_center_distance(X, center, weights, m):
    """Squared Euclidean distance

    Parameters
    -----------
    `X`: shape = [n_sample, m_features]

    `center`: shape = [m_features, ]
    
    `weights`: shape = [m_features]
    
    `m`: float

    Returns
    -----------
    distance:  shape = [n_sample, ]
      n sample's Squared Euclidean distance between certain cluster
    """
    return np.dot((X - center) ** 2, (weights ** m).T)


def kmedian_to_center_distance(X, center, weights, m):
    """Manhattan  distance
    
    Parameters
    -----------
    X: [n_sample, m_features]
    
    center: [m_features,]
    
    weights: [m_features,]
    
    m: float
    
    Returns
    distance:  shape = [n_sample, ]
      n sample's Manhattan distance between certain cluster
    -----------
    
    """
    return np.dot(np.abs(X - center), (weights ** m).T)


def kmeans_center(X):
    """
    
    Parameters
    -----------
    X: [n_sample, m_features]
    """
    return np.mean(X, axis=0)


def kmedian_center(X):
    """
    
    Parameters
    -----------
    X: [n_sample, m_features]
    """
    return np.median(X, axis=0)


class FeatureWeightingKMeans():
    """
    Parameters
    ----------
    
    n_clusters: int
      The number of clusters
            
    weights_: np.ndarray or list, [n_features]
      weight parameter, m >=1 The closer to m is to 1, the closter to hard kmeans.
    
    cluster_method: {'kmeans, kmedian'}
    
    init_center: {'random', 'k-means++', narray}
      init center method 
      
    init_weight: {'random', 'fixed', narray}
    
    m: float, default m=2
      weight parameter, m >=1 The closer to m is to 1, the closter to hard kmeans.
            
    max_iter: int
      Maximum number of iterations of the k-means algorithm for a single run.
            
    tot: float
      Relative tolerance with regards to inertia to declare convergence
            
    random_state : integer or numpy.RandomState, optional
      The generator used to initialize the centers. If an integer is
      given, it fixes the seed. Defaults to the global numpy random
      number generator.
     
    verbose: True or False
      if True, print processing infomation  
            
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
    Coordinates of cluster centers

    labels_ :
      labels of each point
        
    """

    def __init__(self, n_clusters, m=2,
                 cluster_method='k-median',
                 init_weight='random', 
                 init_center='k-means++', 
                 max_iter=300, 
                 random_state=0, 
                 tol=1e-6, 
                 verbose=False):
        self.n_clusters = n_clusters
        assert m >= 1, 'm cannot be less than 1'
        self.m = m
        self.cluster_method = cluster_method
        self.init_weight = init_weight
        self.init_center = init_center
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose

    def _update_label(self, X):
        """
        Fix Center, Weight, Update Label
        """
        n_samples, m_features = X.shape
        assert m_features == len(self.weights_)

        # Choose cluster method 
        if self.cluster_method == 'k-means':
            to_center = kmeans_to_center_distance
        elif self.cluster_method == 'k-median':
            to_center = kmedian_to_center_distance
        else:
            raise ValueError('cluster_method must be kmeans or kmedian')

        # Distance between X and all cluster center
        affiliation = np.zeros((n_samples, self.n_clusters))

        # Calculate distance between data and center i
        for i, center in enumerate(self.cluster_centers_):
            center_i_dist = to_center(X, center, self.weights_, self.m)
            affiliation[:, i] = center_i_dist.T

        # Set label to closest cluster 
        self.labels_ = np.argmin(affiliation, axis=1)

    def _update_center(self, X):
        """
        Fix Label, Weight, Update Center
        """
        centers_old = self.cluster_centers_.copy()

        if self.cluster_method == 'k-means':
            cluster_center = kmeans_center
        elif self.cluster_method == 'k-median':
            cluster_center = kmedian_center
        else:
            raise ValueError('cluster_method must be kmeans or kmedian')

        # Choose data belong to cluster k and
        # Update cluster center with it mean
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            self.cluster_centers_[k] = cluster_center(X[mask])
        
        # check cluster is empty
        if np.isnan(self.cluster_centers_).any():
                raise ValueError('Cluster must have at least one member')
                
        center_shift_total = squared_norm(self.cluster_centers_ - centers_old)
        return center_shift_total

    def _update_weight(self, X):
        """
        Fix Label, Center, Update Weight
        """
        n_samples, m_features = X.shape
        D = np.zeros(m_features)
        
        for feature_id in range(m_features): 
            feature_var = 0        
            for cluster_id in range(self.n_clusters):
                mask = self.labels == cluster_id
                feature_cluster_var = X[mask][:, feature_id] - self.cluster_centers_[cluster_id:, feature_id]
                feature_var += feature_cluster_var
                
        D[feature_id] = feature_var
        beta = 1 / (self.m - 1)
        mask = D != 0
        C = D[mask]
        # weights_ = 1 / (D ** beta * np.sum((1 / D) ** beta))
        weights_ = 1 / (C ** beta * np.sum((1 / C) ** beta))
        D[mask] = weights_
        self.weights_ = D
 
    def fit(self, X, y=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape = [n_samples,]
            Index of the cluster each sample belongs to.
        """
        n_samples, m_features = X.shape

        # variance
        variances = np.mean(np.var(X, 0))
        self.tol *= variances

        # Initialize weight matrix
        if hasattr(self.init_weight, '__array__'):
            print(self.init_weight)
            self.weights_ = self.init_weight
        elif self.init_weight == 'random':
            self.weights_ = _weightmatrix(self.n_clusters, m_features)
        elif self.init_weight == 'fixed':
            self.weights_ = np.ones((self.n_clusters, m_features)) * (1 / m_features)
        else:
            raise Exception('init_weight_must be `random` , `fixed` or numpy.array')
        
        # Initialize center
        # need robust data check  
        if hasattr(self.init_center, '__array__'):
            self.cluster_centers_ = self.init_center
        elif self.init_center == 'k-means++':
            random_state = check_random_state(self.random_state)
            self.cluster_centers_ = _k_init(X=X, n_clusters=self.n_clusters, random_state=random_state)
        elif self.init_center == 'random':
            random_state = check_random_state(self.random_state)
            chosen_ids = random_state.permutation(n_samples)[:self.n_clusters]
            self.cluster_centers_ = X[chosen_ids]
        else:
            raise Exception('init_center must be `random`, `kmeans++` or np.array')

        if self.verbose:
            print('origin center_ \n', self.cluster_centers_)
        
        
        # Iteration
        for i in range(self.max_iter):
        	# update label 
            self._update_label(X)
            # update center
            center_shift_total = self._update_center(X)
            # if weight is fixed continue
            if self.init_weight in ['random', 'fixed']:
            	self._update_weight()

            if self.verbose:
                print('Iteration %i cluster_centers_\n' % i, self.cluster_centers_)
                print('Iteration %i tolerance: ' % i, center_shift_total)
            if center_shift_total < self.tol:
                break

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters  
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape = [n_samples,]
            Index of the cluster each sample belongs to.
        """

        # Check array
        n_samples, dim = X.shape
        m_features = self.cluster_centers_.shape[1]

        if self.cluster_method == 'kmeans':
            to_center = kmeans_to_center_distance
        elif self.cluster_method == 'kmedian':
            to_center = kmedian_to_center_distance
        else:
            raise ValueError('cluster_method must be kmeans or kmedian')

        if self.cluster_centers_ is None:
            self.fit(X)
            return self.labels_
        elif m_features == dim:
            affiliation = np.zeros((n_samples, self.n_clusters))
            # Calculate distance between data and center i
            for i, center in enumerate(self.cluster_centers_):
                center_i_dist = to_center(X, center, self.weights_, self.m)
                if self.verbose:
                    print(center_i_dist)
                affiliation[:, i] = center_i_dist.T

            # Set min distance () be the arbitrary cluster label
            labels_ = np.argmin(affiliation, axis=1)
            return labels_
        else:
            raise ValueError('The features of the X  %i'
                             'does not match the number of clusters %i' 
                             % (dim, m_features))


