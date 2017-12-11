# Finding Influential Institutions in Bibliographic Information Network
- This is the code used in KDD Cup 2016 solution "Finding Influential Institutions in Bibliographic Information Networks".
- This solution achieved 11th position in the contest. Full paper describing the solution is available at https://arxiv.org/abs/1612.08644


## Instructions for running this code
- Download following files in directory **data**:
- Followint columns should be present in each file

    #### SelectedAffiliations.txt
    |1|Affiliation ID|
    |-|--------------|
    |2|Affiliation name|

    #### Authors.txt
    |1|Author ID|
    |-|----------|
    |2|Author name|

    #### Conferences.txt
    |1|	Conference series ID|
    |-|---------------------|
    |2|	Short name (abbreviation)|
    |-|-------------------------|
    |3|	Full name|

    #### ConferenceInstances.txt
    |1|	Conference series ID|
    |2|	Conference instance ID|
    |3|	Short name (abbreviation)|
    |4|	Full name|
    |5|	Location|
    |6|	Official conference URL|
    |7|	Conference start date|
    |8|	Conference end date|
    |9|	Conference abstract registration date|
    |10|	Conference submission deadline date|
    |11|	Conference notification due date|
    |12|	Conference final version due date|

    #### FieldOfStudyHierarchy.txt
    1	Child field of study ID
    2	Child field of study level
    3	Parent field of study ID
    4	Parent field of study level
    5	Confidence

    #### SelectedPapers.txt
    1	Paper ID
    2	Original paper title
    3	Normalized paper title
    4	Paper publish year
    5	Paper publish date 
    6	Paper Document Object Identifier (DOI)
    7	Original venue name
    8	Normalized venue name
    9	Journal ID mapped to venue name
    10	Conference series ID mapped to venue name
    11	Paper rank

    #### PaperAuthorAffiliations.txt
    1	Paper ID
    2	Author ID
    3	Affiliation ID 
    4	Original affiliation name
    5	Normalized affiliation name
    6	Author sequence number

    #### PaperKeywords.txt
    1	Paper ID
    2	Keyword name
    3	Field of study ID mapped to keyword

    #### PaperReferences.txt
    1	Paper ID
    2	Paper reference ID

    -  The code is available in the directory named "Code".
    -  All the code is written in Python 2.7
    -  To run the code, go to the directory "Code" and run the file "main.py".
