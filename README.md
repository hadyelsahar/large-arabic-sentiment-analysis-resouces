# Sentiment Analysis Resouces For Arabic Language

## Overview :

The Repository includes the following :
  - 33K Automatically annotated Reviews in Domains of Movies, Hotels, Restaurants and Products
  - Domain specific lexicons, semi automatically generated from the datasets above (2K total)
  - A total of 615 Experiments over each of the datasets experimenting :
    - Classifiers : Linear SVM, Logistic Regression,  KNN, BNB, SGD training with SVM (Hinge loss and L1 penality) 
    - Sandard Features : TFIDF, Term Count, Term Existence, Delta-TFIDF
    - Lexicon Based Features: domain specific and domain general 
    - Combining features : Lexicon based feature vectors + Standard features 
    - Classification Problems : with neutral class included or not 
    - Balanced or unBalanced Datasets
  - Results of Each of the Experiments


## Dataset Statistics

### Datasets :

####ATT.csv
- Dataset of Attraction Reviews scrapped from TripAdvisor.com 
- 2154 reviews

####HTL.csv
- Dataset of Hotel Reviews scrapped from TripAdvisor.com
- 15572 reviews

####MOV.csv
- Dataset of Movie Reviews scrapped from elcinema.com
- 1524 reviews

####PROD.csv
- Dataset of product reviews scrapped from souq.com
- 4272 reviews

####RES1.csv
- dataset of restaurant reviews scrapped from qaym.com
- 8364 reviews

####RES2.csv
- dataset of restaurant reviews scrapped from tripadvisor.com
- 2642 reviews

####RES.csv
- RES1.csv and RES2.csv combined
- 10970 reviews


## Lexicons
Domain specific lexicons, semi automatically generated from the datasets above (2K total)

|lexicon  | MOV| RES | PROD | HTL | BOOK | Total|
|---------|----|-----|------|-----|------|------| 
| size    | 87 | 734 |  369 |218  |874   | 1913 |


## Requirements

