from collections import defaultdict
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize

conf_ids = defaultdict(str)

confIdMappings = defaultdict(int)
revConfIdMappings = defaultdict(str)
affIdMappings = defaultdict(int)
revAffIdMappings = defaultdict(str)
authorIdMappings = defaultdict(int)
revAuthorIdMappings = defaultdict(str)
keywordMappings = defaultdict(int)
revKeywordMappings = defaultdict(str)
keywordNameMappings = defaultdict(str)
affNameMappings = defaultdict(str)
revAffNameMappings = defaultdict(str)
fosMappings = defaultdict(list)  # Maps every field_of_study to it's field_of_study at level 1

def map_affiliations():
    affFile = open('../data/SelectedAffiliations.txt', 'r')
    affCount = 1
    for line in affFile:
        affId, affName = line.lower().strip().split('\t')
        affNameMappings[affId] = affName
        revAffNameMappings[affName] = affId
        if affIdMappings[affId] == 0:
            affIdMappings[affId] = affCount
            revAffIdMappings[affCount] = affId
            affCount += 1
    affFile.close()

def map_authors():
    authorFile = open('../data/Authors.txt', 'r')
    authorCount = 1
    for line in authorFile:
        authorId, authorName = line.strip().split('\t')
        if authorIdMappings[authorId] == 0:
            authorIdMappings[authorId] = authorCount
            revAuthorIdMappings[authorCount] = authorId
            authorCount += 1
    authorFile.close()

def map_confs():
    confFile = open('../data/Conferences.txt', 'r')
    for line in confFile:
        line = line.strip().split('\t')
        conf_ids[line[1]] = line[0]
    confFile.close()

# Create mappings from a FOS to it's (parent_FOS, parent_level)
def map_fos():
    fosFile = open('../data/FieldOfStudyHierarchy.txt', 'r')
    for line in fosFile:
        line = line.lower().strip().split('\t')
        cLevel = int(line[1][-1])
        pLevel = int(line[3][-1])
        if cLevel >= 2 and pLevel >= 1:
            if fosMappings[line[0]] == [] or fosMappings[line[0]][1] != 1:
                fosMappings[line[0]] = [line[2], pLevel]
    fosFile.close()
    
# Creates mappings from FOS ids to indices
def map_keywords():
    keywordFile = open('../data/PaperKeywords.txt', 'r')
    keywordCount = 1
    for line in keywordFile:
        line = line.strip().split('\t')
        if fosMappings[line[2]] == []:
            fosMappings[line[2]] = [line[2], 1]
        keyword = fosMappings[line[2]][0]
        if keywordMappings[keyword] == 0:
            keywordMappings[keyword] = keywordCount
            revKeywordMappings[keywordCount] = keyword
            keywordNameMappings[keywordCount] = line[1]
            keywordCount += 1
    keywordFile.close()

def create_mappings():
    map_affiliations()
    map_authors()
    map_confs()
    map_fos()
    map_keywords()

class YearData:
    def __init__(self, year):
        self.year = year
        self.paperIdMappings = defaultdict(int)
        self.revPaperIdMappings = defaultdict(str)

    def map_papers_confs(self):
        paperFile = open('../data/SelectedPapers.txt', 'r')
        paperCount = 1
        confCount = 1
        for line in paperFile:
            line = line.strip().split('\t')
            if int(line[2]) == self.year and self.paperIdMappings[line[0]] == 0:
                self.paperIdMappings[line[0]] = paperCount
                self.revPaperIdMappings[paperCount] = line[0]
                paperCount += 1
            if confIdMappings[line[3]] == 0:
                confIdMappings[line[3]] = confCount
                revConfIdMappings[confCount] = line[3]
                confCount += 1
        paperFile.close()
        
    def load_paper_conf_mat(self):
        numPapers = len(self.paperIdMappings)
        numConfs = len(confIdMappings)
        M_PC = lil_matrix((numPapers+1, numConfs+1))
        paperFile = open('../data/SelectedPapers.txt', 'r')
        for line in paperFile:
            line = line.strip().split('\t')
            if int(line[2]) == self.year:
                pId = self.paperIdMappings[line[0]]
                confId = confIdMappings[line[3]]
                M_PC[pId,confId] = 1
        paperFile.close()
        return M_PC
        
    def load_paper_keyword_mat(self):
        numPapers = len(self.paperIdMappings)
        numKeywords = len(keywordMappings)
        M_PK = lil_matrix((numPapers+1, numKeywords+1))
        keywordFile = open('../data/PaperKeywords.txt', 'r')
        for line in keywordFile:
            line = line.strip().split('\t')
            keyword = fosMappings[line[2]][0]
            pid = self.paperIdMappings[line[0]]
            keywordId = keywordMappings[keyword]
            M_PK[pid, keywordId] = 1
        keywordFile.close()
        return M_PK
        
    # Returns three matrices (M_pa, M_paff, M_aaff)
    def load_paper_author_conf_aff_mats(self, M_PC):
        prevAuthors = defaultdict(set)
        numPapers = len(self.paperIdMappings)
        numAffs = len(affIdMappings)
        numAuthors = len(authorIdMappings)
        numConfs = len(confIdMappings)
        M_PA = lil_matrix((numPapers+1, numAuthors+1))
        M_AO = lil_matrix((numAuthors+1, numAffs+1))
        M_COAcount = lil_matrix((numConfs+1, numAffs+1))
        M_PC_dense = M_PC.toarray()
        paFile = open('../data/PaperAuthorAffiliations.txt', 'r')
        for line in paFile:
            line = line.strip().split('\t')
            if line[0] in self.paperIdMappings:
                pId = self.paperIdMappings[line[0]]
            else:
                continue
            if line[1] in authorIdMappings:
                aId = authorIdMappings[line[1]]
            else:
                continue
            if line[2] in affIdMappings:
                affId = affIdMappings[line[2]]
            else:
                continue
            M_PA[pId,aId] = 1
            M_AO[aId,affId] = 1
            confId = M_PC_dense[pId].argmax()
            if aId not in prevAuthors[confId]:
                prevAuthors[confId].add(aId)
                M_COAcount[confId,affId] = M_COAcount[confId,affId] + 1
        paFile.close()
        M_PA = normalize(M_PA, norm='l1', axis=1)
        M_AO = normalize(M_AO, norm='l1', axis=1)
        return (M_PA, M_AO, M_COAcount)
        
def load_all_mats(data):
    M_PC = data.load_paper_conf_mat()
    M_PA, M_AO, M_COAcount = data.load_paper_author_conf_aff_mats(M_PC)
    M_PK = data.load_paper_keyword_mat()
    M_AA = M_PA.transpose() * M_PA
    M_CK = M_PC.transpose() * M_PK
    M_PO = M_PA * M_AO
    M_OC = M_PO.transpose() * M_PC
    M_CA = M_PC.transpose() * M_PA
    M_CC = M_CA * M_CA.transpose()
    M_CC = normalize(M_CC, norm='l1', axis=0)
    M_OC = normalize(M_OC, norm='l1', axis=0)
    M_OK = M_PO.transpose() * M_PK
    M_OK = normalize(M_OK, norm='l1', axis=1)
    return (M_PC,M_PA,M_AO,M_COAcount,M_PK,M_AA,M_CK,M_PO,M_OC,M_CC,M_OK)
