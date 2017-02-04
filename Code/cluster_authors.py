from collections import defaultdict
from scipy.sparse import lil_matrix
import numpy as np

def clusterAuthors(K, authorIdMappings):
    citationCounts = defaultdict(int)
    refsFile = open('../data/PaperReferences.txt', 'r')
    for line in refsFile:
        line = line.lower().strip().split('\t')
        pid = line[0]
        citationCounts[pid] = citationCounts[pid] + 1
    refsFile.close()
    
    authorCitationCounts = defaultdict(int)
    paFile = open('../data/PaperAuthorAffiliations.txt', 'r')
    for line in paFile:
        line = line.lower().strip().split('\t')
        pid = line[0]
        aid = line[1]
        authorCitationCounts[aid] = authorCitationCounts[aid] + citationCounts[pid]
    paFile.close()
    
    counts = np.zeros((len(authorIdMappings)+1,))
    for k,v in authorCitationCounts.items():
        aid = authorIdMappings[k]
        counts[aid] = v
        
    indices = counts.argsort()
    if len(counts) % K == 0:
        numElements = len(counts) / K
    else:
        numElements = len(counts) / K + 1
    groups = []
    groups.append(indices[-numElements:])
    for i in range(1,K):
        groups.append(indices[-numElements*(i+1):-numElements*i])
    
    M_AAclus = lil_matrix((len(authorIdMappings)+1,K))
    for i in range(K):
        for j in range(len(groups[i])):
            M_AAclus[groups[i][j],i] += 1
            
    return M_AAclus
