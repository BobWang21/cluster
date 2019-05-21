#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Impliment Davies–Bouldin index
The Less The Better
https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
@author: wangbao
"""
import numpy as np


def dbi(center, data, label):
    """
    center: cluster center (n_cluster, n_feature)
    data: (n, n_feature)
    label (n, )
    """
    cluster_num = center.shape[0]
    
    S = np.zeros(cluster_num) 
    for cluster_idx in range(cluster_num):
        mask = (label == cluster_idx)
        cluster_data = data[mask]
        num = cluster_data.shape[0]
        S[cluster_idx] = np.linalg.norm(cluster_data - center[cluster_idx], 2) / num
    
    D = np.zeros(cluster_num)
    for center_i in range(cluster_num):
        d_max = -np.inf
        for center_j in range(cluster_num):
            if center_i == center_j:
                continue
            else:
                m = np.linalg.norm(center[center_i] - center[center_j], 2)
                # print(m)
                d = (S[center_i] + S[center_j]) / m
                d_max = max(d, d_max)
        D[center_i] = d_max
    averge_D = np.mean(D)
    return averge_D