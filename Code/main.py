# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:26:13 2016

@author: anubhav.gupta
"""

from IO import *
from collections import defaultdict
from cluster_authors import clusterAuthors
from algorithms import *
from numpy.linalg import norm
from scipy.sparse import vstack
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor
import os
import cPickle
from numpy.linalg import pinv
import matplotlib.pyplot as plt

""" Some notation:

A - Author
P - Paper
O - Organization/Institution
K - FOS
C - Conference

"""

valid_confs = ['sigir','sigmod','sigcomm','kdd','icml','fse','mobicom','mm']

def create_matrix(data, M_AAclus): 
    M_PC,M_PA,M_AO,M_COAcount,M_PK,M_AA,M_CK,M_PO,M_OC,M_CC,M_OK = load_all_mats(data)
    M_CA = M_PC.transpose() * M_PA
    M_OA = M_AO.transpose()
    M_CAclus = M_CA * M_AAclus
    M_OAclus = M_OA * M_AAclus
    m = M_CA.shape[0]
    M_OAclus_1 = M_OAclus[1:].multiply(M_CAclus[1])
    M_OK_1 = M_OK[1:].multiply(M_CK[1])
    M = hstack([M_OAclus_1,M_OK_1,M_COAcount[1,1:].transpose()]).tolil()
    for i in range(2,m):
        M_OAclus_i = M_OAclus[1:].multiply(M_CAclus[i])
        M_OK_i = M_OK[1:].multiply(M_CK[i])
        res = hstack([M_OAclus_i,M_OK_i,M_COAcount[i,1:].transpose()]).tolil()
        M = vstack([M, res]).tolil()
    M_CO = M_OC.tolil().transpose()
    return (M, M_CO[1:,1:].reshape((np.product(M_CO[1:,1:].shape),1)))

if __name__ == '__main__':
    serialization_dir = './serialize_data500/'
    directory = os.path.dirname(serialization_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    all_w = [float(x)/20 for x in range(0,21)]
    all_scores = defaultdict(list)
    
    print 'Creating some mappings for faster processing...'
    create_mappings()
    years = range(2011,2016)
    dataMat = [[] for x in xrange(5)]
    scoresVec = [[] for x in xrange(5)]
    
    print 'Clustering Authors...'
    M_AAclus = clusterAuthors(500, authorIdMappings)
    for year in years:
        print 'Creating data matrix for year ' + str(year) + '...'
        data = YearData(year)
        data.map_papers_confs()
        fileName = serialization_dir + 'data_mat_' + str(year)
        rFileName = serialization_dir + 'scores_mat_' + str(year)
        if os.path.isfile(fileName) and os.path.isfile(rFileName):
            print 'Loading data matrix from serialization file'
            dataMat[year-2011] = cPickle.load(open(fileName, 'rb'))
            scoresVec[year-2011] = cPickle.load(open(rFileName, 'rb'))
        else:
            print 'Serialization files for year ' + str(year) + ' don\'t exist.'
            print 'Creating the serialization files...'
            M, ranks = create_matrix(data, M_AAclus)
            dataMat[year-2011] = M.copy()
            scoresVec[year-2011] = ranks.copy()
            cPickle.dump(dataMat[year-2011], open(fileName, 'wb'))
            cPickle.dump(scoresVec[year-2011], open(rFileName, 'wb'))
    
    M_11, M_12, M_13, M_14, M_15 = dataMat[0], dataMat[1], dataMat[2], dataMat[3], dataMat[4]
    r_11, r_12, r_13, r_14, r_15 = scoresVec[0], scoresVec[1], scoresVec[2], scoresVec[3], scoresVec[4]
    
    model_15 = None
    model_16 = None
    lamda = 200
    
    outputFile = open('submission.tsv', 'wb')
    
    for conf in valid_confs:
        all_scores[conf] = [0.0 for _ in range(3)]
        
        print 'Computing the data matrix for year 2015 of conference ' + str(conf)
        cid = conf_ids[conf]
        idx = confIdMappings[cid]
        A = M_14[(idx-1)*741:idx*741]
        B = M_13[(idx-1)*741:idx*741]
        C = M_12[(idx-1)*741:idx*741]
        D = M_11[(idx-1)*741:idx*741]
        MM = M_15[(idx-1)*741:idx*741]
        AA = A.reshape((1,np.product(A.shape)))
        BB = B.reshape((1,np.product(B.shape)))
        CC = C.reshape((1,np.product(C.shape)))
        DD = D.reshape((1,np.product(D.shape)))
        X = vstack([AA,BB,CC]).transpose()
        M_hat = MM.reshape((np.product(MM.shape),1))
        XX = X.toarray()
        invX = pinv(XX)
        w = invX * M_hat
        
        X = vstack([BB,CC,DD]).transpose()
        M_hat = A.reshape((np.product(A.shape),1))
        w = (pinv(X.transpose().dot(X) + lamda*np.eye(w.shape[0]))
            .dot(X.transpose().dot(M_hat) + lamda*w))
        w = np.array(w)
        
        M_15_test = w[0][0]*A + w[1][0]*B + w[2][0]*C
        M_16_test = w[0][0]*MM + w[1][0]*A + w[2][0]*B
        
        print 'Fitting RF Regression for 2015'
        X = vstack([M_11,M_12,M_13,M_14]).tolil()
        y = vstack([r_11,r_12,r_13,r_14]).tolil()
        y = y.toarray().transpose()[0]
        if model_15 == None:
            model_15 = RandomForestRegressor(max_depth=50)
            model_15.fit(X,y)
        y_pred_15 = model_15.predict(M_15_test)
        y_pred_15[y_pred_15 < 0] = 0
        
        print 'Fitting RF Regression for 2016'
        X = vstack([M_11,M_12,M_13,M_14,M_15]).tolil()
        y = vstack([r_11,r_12,r_13,r_14,r_15]).tolil()
        y = y.toarray().transpose()[0]
        if model_16 == None:
            model_16 = RandomForestRegressor(max_depth=50)
            model_16.fit(X,y)
        y_pred_16 = model_16.predict(M_16_test)
        y_pred_16[y_pred_16 < 0] = 0
        
        ranks_11 = r_11[(idx-1)*741:idx*741]
        ranks_12 = r_12[(idx-1)*741:idx*741]
        ranks_13 = r_13[(idx-1)*741:idx*741]
        ranks_14 = r_14[(idx-1)*741:idx*741]
        ranks_15 = r_15[(idx-1)*741:idx*741]
        ranks_11 = ranks_11.toarray().transpose()[0]
        ranks_12 = ranks_12.toarray().transpose()[0]
        ranks_13 = ranks_13.toarray().transpose()[0]
        ranks_14 = ranks_14.toarray().transpose()[0]
        ranks_15 = ranks_15.toarray().transpose()[0]
        
        max_score = 0
        for w in all_w:
            w1, w2, w3 = 1, w, w**2
            ranking_scores = w1*ranks_13 + w2*ranks_12 + w3*ranks_11
            ranking_scores = ranking_scores / norm(ranking_scores, 1)
            ranks = ranking_scores.copy()
            score = ndcg(ranks, ranks_14)
            if score > max_score:
                max_score = score
                w_opt = w
        scores2_15 = w1*ranks_14 + w2*ranks_13 + w3*ranks_12
        scores2_15 = scores2_15 / norm(scores2_15, 1)
        
        print
        print 'Calculating the baseline scores...'
        print
        print '============================================'
        print '             Baseline for ' + conf.upper()
        print '============================================'
        print '%-10s %-10s %-10s %-10s' %('Year', 'NDCG@10', 'NDCG@20', 'NDCG@30')
        print '--------------------------------------------'
        ndcg_10_15 = ndcg(ranks_15, ranks_14, 10)
        ndcg_20_15 = ndcg(ranks_15, ranks_14, 20)
        ndcg_30_15 = ndcg(ranks_15, ranks_14, 30)
        print '%-10s %-10.4f %-10.4f %-10.4f' %('2015', ndcg_10_15, ndcg_20_15, ndcg_30_15)
        print '============================================'
        all_scores[conf][0] = ndcg_20_15
        
        print
        print 'Results for RankIns2...'
        scores_15 = y_pred_15.copy()
        print
        print '============================================'
        print '        Results of LR for ' + conf.upper()
        print '============================================'
        print '%-10s %-10s %-10s %-10s' %('Year', 'NDCG@10', 'NDCG@20', 'NDCG@30')
        print '--------------------------------------------'
        ndcg_10_15 = ndcg(ranks_15, scores_15, 10)
        ndcg_20_15 = ndcg(ranks_15, scores_15, 20)
        ndcg_30_15 = ndcg(ranks_15, scores_15, 30)
        print '%-10s %-10.4f %-10.4f %-10.4f' %('2015', ndcg_10_15, ndcg_20_15, ndcg_30_15)
        print '============================================'
        all_scores[conf][2] = ndcg_20_15
        
        print
        print 'Results for RankIns1...'
        print
        print '============================================'
        print '        Results of LR for ' + conf.upper()
        print '============================================'
        print '%-10s %-10s %-10s %-10s' %('Year', 'NDCG@10', 'NDCG@20', 'NDCG@30')
        print '--------------------------------------------'
        ndcg_10_15 = ndcg(ranks_15, scores2_15, 10)
        ndcg_20_15 = ndcg(ranks_15, scores2_15, 20)
        ndcg_30_15 = ndcg(ranks_15, scores2_15, 30)
        print '%-10s %-10.4f %-10.4f %-10.4f' %('2015', ndcg_10_15, ndcg_20_15, ndcg_30_15)
        print '============================================'
        all_scores[conf][1] = ndcg_20_15
        
        result = []
        for i in range(y_pred_16.shape[0]):
            affId = revAffIdMappings[i+1]
            cid = conf_ids[conf]
            result.append(cid.upper()+'\t'+affId.upper()+'\t'+'{0:.12f}'.format(y_pred_16[i]))
        outputFile.write('\n'.join(result)+'\n')
    outputFile.close()


    N = 8
    ind = np.arange(N)+0.15  # the x locations for the groups
    width = 0.10      # the width of the bars
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    yvals1 = [all_scores[conf][0] for conf in valid_confs]
    rects1 = ax.bar(ind, yvals1, width, color='r', align='center')
    yvals2 = [all_scores[conf][1] for conf in valid_confs]
    rects2 = ax.bar(ind+width, yvals2, width, color='y', align='center')
    yvals3 = [all_scores[conf][2] for conf in valid_confs]
    rects3 = ax.bar(ind+width*2, yvals3, width, color='b', align='center')
    ax.set_ylabel('NDCG@20 (For 2015)')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(('SIGIR','SIGMOD','SIGCOMM','KDD',
                        'ICML','FSE','MobiCom','MM'),
                        horizontalalignment = 'center')                        
    ax.legend((rects1[0],rects2[0],rects3[0]), ('PreviousYear', 'RankIns1', 'RankIns2'))
    plt.show()
