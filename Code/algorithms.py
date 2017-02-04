import numpy as np
from IO import *

""" Find top-20 organizations with paper counts in conference conf. """
def topOrganizations(M_OC, conf):
    orgList = np.array(M_OC[:,conf].todense()).reshape(-1,)
    sortedList = np.sort(orgList)[::-1]
    temp = np.argsort(orgList)
    ranks = np.empty(len(orgList))
    ranks[temp] = len(orgList) - np.arange(len(orgList))
    temp2 = ranks.argsort()
    for i in range(30):
        print '{:>50}'.format(affNameMappings[revAffIdMappings[temp2[i]]]), sortedList[i]

""" Compute Normalized Discounted Cumulative Gain (NDCG@k)
rel => Vector of true relevance scores
prob => Vector of predicted probabilities
"""
def ndcg(rel, prob, k=20):    
    # Let's compute DCG
    temp = np.argsort(prob)
    predRanks = np.empty(len(prob))
    predRanks[temp] = len(prob) - np.arange(len(prob))
    i_s = np.argsort(predRanks)
    logRanks = np.log2(predRanks+1)
    dcg = np.sum(rel[i_s][:k] / logRanks[i_s][:k])
    
    # Let's compute IDCG
    numerator = np.sort(rel)[::-1]
    denominator = np.sort(logRanks)
    idcg = np.sum(numerator[:k] / denominator[:k])
    return dcg / idcg
