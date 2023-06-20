import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
import numpy as np
import re
import string
import sklearn.ensemble
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
import time

stop_words=stopwords.words('english')

#function to non-usable characters from text
def text_preproc(x):
    x = x.lower()
    x = ' '.join([word for word in x.split(' ') if word not in stop_words])
    x = x.encode('ascii', 'ignore').decode()
    x = re.sub(r'https*\S+', ' ', x)
    x = re.sub(r'@\S+', ' ', x)
    x = re.sub(r'#\S+', ' ', x)
    x = re.sub(r'\'\w+', '', x)
    x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
    x = re.sub(r'\w*\d+\w*', '', x)
    x = re.sub(r'\s{2,}', ' ', x)

    return x

#class to read in data and preprocess
class data_processor:
    
    #initialize with dataframe file and category for x (train text) and y (website sentiment value)
    def __init__(self, df, x_column, y_column):
        self.df = df
        self.x = self.df[x_column].tolist()
        self.y_column = y_column
     
    #get processed features (uses text preproc) and sparse output vector
    def run_preprocessing(self):
        self.y = []
        y_tmp=pd.get_dummies(self.y_column)
        for i in range(y_tmp.shape[0]):
            tmp_list=y_tmp.iloc[i,:].tolist()
            idx=tmp_list.index(1)
            self.y.append(idx)

        for i in range(len(self.x)):
            self.x[i]=text_preproc(self.x[i])
        
        vectorizer = TfidfVectorizer(max_features=5000, min_df=5,max_df=0.8, stop_words=stopwords.words('english'))
        self.processed_x=vectorizer.fit_transform(self.x).toarray()
        

    #return data variables for use
    def get_data(self):
        return self.processed_x, self.y


#class to generate and run classifiers
class Classifiers:

    def __init__(self, classifier_list):
        if 'rfc' in classifier_list or 'RFC' in classifier_list:
            self.RFC = sklearn.ensemble.RandomForestClassifier()
        if 'mnb' in classifier_list or 'MNB' in classifier_list:
            self.MNB = MultinomialNB()
        if 'xgb' in classifier_list or 'XGB' in classifier_list:
            self.XGB = xgb.XGBClassifier(objective="logistic")
            
    #cross vals for different classifiers, params should be in the syntax of a dictionary, options to print time elapsed
    def cv_rfc(self, x_data, y_data, params, cv_split, timed = False):
        if timed:
            start_time = time.time()
        self.cv_rfc=GridSearchCV(self.RFC, params, scoring = 'accuracy', cv = cv_split)
        self.cv_rfc.fit(x_data, y_data)
        if timed:
            print('time_elapsed for rfc run: ', time.time()-start_time,' seconds')
        
    def cv_mnb(self, x_data, y_data, params, cv_split, timed = False):
        if timed:
            start_time = time.time()
        self.cv_mnb=GridSearchCV(self.MNB, params, cv = cv_split)
        self.cv_mnb.fit(x_data, y_data)
        if timed:
            print('time_elapsed for mnb run: ', time.time()-start_time,' seconds')
            
    def cv_xgb(self, x_data, y_data, params, cv_split, timed = False):
        if timed:
            start_time = time.time()
        self.cv_xgb=GridSearchCV(self.XGB, params, cv = cv_split)
        self.cv_xgb.fit(x_data, y_data)
        if timed:
            print('time_elapsed for xgb run: ', time.time()-start_time,' seconds')
            
    #return scores
    def get_scores_rfc(self):
        return self.cv_rfc.cv_results_
    
    def get_scores_mnb(self):
        return self.cv_mnb.cv_results_
    
    def get_scores_xgb(self):
        return self.cv_xgb.cv_results_
    
    def get_best_score_rfc(self):
        return self.cv_rfc.best_score_, self.cv_rfc.best_params_
    
    def get_best_score_mnb(self):
        return self.cv_mnb.best_score_, self.cv_mnb.best_params_
    
    def get_best_score_xgb(self):
        return self.cv_xgb.best_score_, self.cv_xgb.best_params_