from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from itertools import groupby as g
import numpy as np
import math 


class InstanceBasedLearner(SupervisedLearner):
    features = []
    labels = []
    orig_features = []
    k = 3
    continuous = False
    weighted = False
    random_batch = True

    def __init__(self):
        pass

    def train(self, features, labels):
        self.orig_features = features

        if features.value_count <= 0:
            self.continuous = True
        else:
            self.continuous = False
        if self.random_batch:
            self.features, self.labels, self.orig_features = self.random_reduce_data(features.data, labels.data)
        else :
            self.features = features.data
            self.labels = labels.data
            self.orig_features = features

    def random_reduce_data(self, features, labels):
        len_features = len(features)
        step = 1 #int(len(features)/2000)
        new_feat = []
        new_label = []
        new_orig_feat = Matrix()
        for i in range(0, len_features, step):
            new_feat.append(features[i])
            new_label.append(labels[i])
            new_orig_feat.add(self.orig_features, i, 0, 1)
            #if len(new_feat) >= 2000:
            #    break
        return new_feat, new_label, new_orig_feat
            
    def predict(self, features, labels):
        del labels[:]

        eu_dist = []
        # for every instance calculate the euclidian distance 
        for i in range(len(self.features)):
            eu_dist.append(self.find_eu_dist(self.features, i, features))
            
        voting_data = self.find_voting_data(eu_dist)
        
        # vote
        votes = []
        # uses each stored index in the voting data array of closest objects.
        for min_dist in voting_data: 
            votes.append(self.labels[min_dist])
        # putting votes in a set improves time complexity
        if self.continuous:
            # mean of votes
            mean = 0
            for i in votes:
                mean += i
            mean = mean / len(votes)
            prediction = mean
        else:
            prediction = max(votes, key=votes.count)
        labels += prediction

    def find_eu_dist(self, features, i, prediction):
        # not actually uclidean distance, actually is the Manhatan distance
        # since this program ran very slow I switched to a less computationally
        # heavy distance metric
        
        eu_dist = 0.0
        for j in range(len(features[i])):
            # weighted option
            distance = 0.0
            if self.weighted:
                if self.orig_features.value_count(j) > 0:
                    self.nominal_data(features[i][j], prediction[j], eu_dist)
                else: 
                    distance =  abs(features[i][j] - prediction[j])
                    if not distance == 0.0:
                        weight = 1 / distance
                        eu_dist += weight * distance
                    else:
                        eu_dist += distance
            else:
                if self.orig_features.value_count(i) > 0:
                    self.nominal_data(features[i][j], prediction[j], eu_dist)
                else:
                    eu_dist += abs(features[i][j] - prediction[j])
        return eu_dist
        #return distances
        
    def find_voting_data(self, eu_dist):
        voting_data = []
        for i in range(self.k):
            min_num = min(eu_dist)
            min_index = eu_dist.index(min(eu_dist))
            voting_data.append(min_index)
            eu_dist[min_index] = float("inf")
        return voting_data

    def nominal_data(self, feature, predict,  eu_dist):
        #see page 5 of linked paper in content
        if not feature == predict:
            eu_dist = eu_dist + 1
        elif predict == float("inf"):
            eu_dist = eu_dist + 1
        
