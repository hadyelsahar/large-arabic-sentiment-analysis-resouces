# -*- coding: utf-8 -*-

import sys
import argparse
import csv

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import cross_validation
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

from classes.Document import *
from classes.Utils import * 
from classes.LexiconVectorizer import * 
from classes.DeltaTfidf import * 


valid_datasets = ["SUQ","QYM","TRH","TRR","MOV","LABR","RES"]
valid_datasets = ["SUQ","TRH","MOV","LABR","RES"]

parser = argparse.ArgumentParser(description='Sentiment classification Experiments')
parser.add_argument('-d','--dataset',
    help='which dataset to run experiment on',required=False)
parser.add_argument('-o','--output', help='ouput file name',required=True)
args = parser.parse_args()
if args.dataset is None : 
    datasets = valid_datasets
else :
    if args.dataset in valid_datasets:
        datasets = [args.dataset]
    else : 
        print " only available datasets are " + str(valid_datasets)
        sys.exit()

for dname in datasets : 

    vectorizers  = {
                    "tfidf" : TfidfVectorizer(
                            tokenizer=TreebankWordTokenizer().tokenize,
                            ngram_range=(1,2),norm="l1",
                            preprocessor = Document.preprocess
                        ),
                    "count" : CountVectorizer(
                            tokenizer=TreebankWordTokenizer().tokenize,
                            ngram_range=(1,2),
                            preprocessor = Document.preprocess
                        ),
                    "lex-domain" : LexiconVectorizer(
                            lexfile='lexicon/%s_lex.csv'%dname,
                            polarity = True,
                            weightedcount = True,
                            preprocessor = Document.preprocess
                        ),
                    "lex-all" : LexiconVectorizer(
                            lexfile='lexicon/ALL_lex.csv',
                            polarity = True,
                            weightedcount = True,
                            preprocessor = Document.preprocess
                        ),
                    "delta-tfidf" : DeltaTfidf(                            
                            tokenizer = TreebankWordTokenizer().tokenize,
                            preprocessor = Document.preprocess
                        )
    }

    kfolds = {                
                "CV_unBalanced_2C" : create_dataset(dname, 
                    CV = True, neutral = False, balanced = False, n_folds = 5
                    ),
                "CV_unBalanced_3C" : create_dataset(dname, 
                    CV = True, neutral = True, balanced = False, n_folds = 5
                    ),
                "CV_Balanced_2C" : create_dataset(dname, 
                    CV = True, neutral = False, balanced = True, n_folds = 5
                    ),
                "CV_Balanced_3C" : create_dataset(dname, 
                    CV = True, neutral = True, balanced = True, n_folds = 5
                    ),
                "Split_unBalanced_2C" : create_dataset(dname, 
                    CV = False, neutral = False, balanced = False, n_folds = 5
                    ),
                "Split_Balanced_2C" : create_dataset(dname, 
                    CV = False, neutral = False, balanced = True, n_folds = 5
                    ),
                "Split_unBalanced_3C" : create_dataset(dname, 
                    CV = False, neutral = True, balanced = False, n_folds = 5
                    ),
                "Split_Balanced_3C" : create_dataset(dname, 
                    CV = False, neutral = True, balanced = True, n_folds = 5
                    )
    }

    
    classifiers = {
                "svm": LinearSVC(penalty="l1", dual=False),
                "svm_cv": GridSearchCV(
                    LinearSVC(penalty="l1", dual=False),
                    [{'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}] #range of C coefficients to try
                    )
                "LREG": LogisticRegression(penalty="l1", dual=False),
                "BernoulliNB" : BernoulliNB(alpha=.01),                
                "SGD" : SGDClassifier(loss="hinge", penalty="l1"),
                "KNN" : KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    }

    #Feature Building
    features = {
                "lex-domain" : FeatureUnion([
                        ("lex-domain", vectorizers["lex-domain"])]
                        ),
                "lex-all" : FeatureUnion([
                        ("lex-all", vectorizers["lex-all"])]
                        ),
                "tfidf" : FeatureUnion([
                        ("tfidf", vectorizers["tfidf"])]
                        ),
                "delta-tfidf" : FeatureUnion([
                        ("delta-tfidf", vectorizers["delta-tfidf"])]
                        ),
                "count" : FeatureUnion([
                        ("count", vectorizers["count"])]
                        ),
                "tfidf_lex-domain" : FeatureUnion([
                        ("lex-domain", vectorizers["lex-domain"]),
                        ("tfidf", vectorizers["tfidf"])]
                        ),
                "delta-tfidf_lex-domain" : FeatureUnion([
                        ("lex-domain", vectorizers["lex-domain"]),
                        ("delta-tfidf", vectorizers["delta-tfidf"])]
                        ),
                "tfidf_lex-all" : FeatureUnion([
                        ("lex-all", vectorizers["lex-all"]),
                        ("tfidf", vectorizers["tfidf"])]
                        ),
                "delta-tfidf_lex-all" : FeatureUnion([
                        ("lex-all", vectorizers["lex-all"]),
                        ("delta-tfidf", vectorizers["delta-tfidf"])]
                        ),
                "count_lex-domain" : FeatureUnion([
                        ("lex-domain", vectorizers["lex-domain"]),
                        ("count", vectorizers["count"])]
                        ),
                "count_lex-all" : FeatureUnion([
                        ("lex-all", vectorizers["lex-all"]),
                        ("count", vectorizers["count"])]
                        )
    }

    fout = open(args.output,"w")
    writer = csv.writer(fout)

    for fold_name,fold in kfolds.items():
        for fvector_name,fvector in features.items():
            for clf_name, clf in classifiers.items():
            
                print "# %s\t%s\t%s\t%s"%(dname, fold_name, fvector_name, clf_name)

                pipeline = Pipeline([
                                ('features', fvector), 
                                # ('select',selector), 
                                ('classifier', clf)])
            
                fold_metrics = []
                total_accuracy = [] 
                for (X_train,y_train,X_test,y_test) in fold:

                    pipeline.fit(X_train,y_train)
                    pred = pipeline.predict(X_test)

                    #metrics of each class
                    m1 = np.array(precision_recall_fscore_support(y_test, pred))
                    #average of all classes
                    m2 = np.array(precision_recall_fscore_support(y_test, pred, 
                        average = "micro"))                    
                    #sum of all supports not average
                    m2[3] = np.sum(m1[3,:])
                    fold_metrics.append(np.c_[m1,m2])    
                    total_accuracy.append(accuracy_score(y_test, pred))

                # average all the metrics of all cross folds
                metrics = np.mean(np.array(fold_metrics),axis = 0)                
                #print metrics report
                x = pd.DataFrame(metrics).transpose()
                ic3 = ["neg","neutral","pos","Average/Total"]
                ic2 = ["neg","pos","Average/Total"]
                x.index =  ic3 if len(metrics[0]) == 4 else ic2                 
                x.columns = ["precision","recall","fscore","support"]                
                
                fscore = x["fscore"]["Average/Total"]
                accuracy = np.mean(total_accuracy)

                print x 
                print "total_accuracy = %s"%accuracy
                print "\n"

                #log average fscore/accuracy to a file 

                writer.writerow([dname, 
                    fold_name,
                    fvector_name,
                    clf_name,                    
                    fscore,
                    accuracy
                    ])

    fout.close()


                




