import ast
import operator
import os
import pickle
import unittest
from collections import Counter

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

import main
from main import preprocess, build

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


class TestSearchQuery(unittest.TestCase):

    def test_preprocess(self):
        doc = 'Preprocess this string and output array of tokens'
        assert preprocess(doc) == ['preprocess', 'string', 'output', 'array', 'token']

    # Given a string that has no appearance in the tf-idf model, the output query_rank table should present null values
    def test_rank_documents_empty(self):
        corpus = pd.read_csv('../dataset/articles1.csv', index_col=0, usecols=[1, 2, 3, 4, 9], header=0)
        if not os.path.isfile('tfidf.joblib'):
            tfidf_model = build(corpus)
        else:
            tfidf_model = joblib.load('tfidf.joblib')
        query = 'nekrjbkt'
        query_rank = main.rank_documents(tfidf_model, query)
        total_score = query_rank['Ranking'].sum()
        assert abs(total_score - 0.0) <= 1e-6

    # Given a string that has appearance in the tf-idf model, the output query_rank table should present positive values
    def test_rank_documents(self):
        corpus = pd.read_csv('../dataset/articles1.csv', index_col=0, usecols=[1, 2, 3, 4, 9], header=0)
        if not os.path.isfile('tfidf.joblib'):
            tfidf_model = build(corpus)
        else:
            tfidf_model = joblib.load('tfidf.joblib')
        query = 'trump'
        query_rank = main.rank_documents(tfidf_model, query)
        total_score = query_rank['Ranking'].sum()
        assert total_score > 0

    # Given a string that has appearance in the model, the output should entail:
    # positive number of total matches
    # positive score for the top <threshold> matches
    def test_search_query(self):
        corpus = pd.read_csv('../dataset/articles1.csv', index_col=0, usecols=[1, 2, 3, 4, 9], header=0)
        if not os.path.isfile('tfidf.joblib'):
            tfidf_model = build(corpus)
        else:
            tfidf_model = joblib.load('tfidf.joblib')
        query = 'trump'
        threshold = 20

        total_n_matches, query_rank, top_match_IDs = main.search_query(tfidf_model, query, threshold)
        assert total_n_matches > 0
        for match in top_match_IDs:
            assert query_rank.loc[match].values[0] > 0


    # Given a string that has no appearance in the model, the output should entail:
    # null number of total matches
    # null score for the top <threshold> matches
    def test_search_query_empty(self):
        corpus = pd.read_csv('../dataset/articles1.csv', index_col=0, usecols=[1, 2, 3, 4, 9], header=0)
        if not os.path.isfile('tfidf.joblib'):
            tfidf_model = build(corpus)
        else:
            tfidf_model = joblib.load('tfidf.joblib')
        query = 'wrgrtejrut'
        threshold = 20

        total_n_matches, query_rank, top_match_IDs = main.search_query(tfidf_model, query, threshold)
        assert total_n_matches == 0
        for match in top_match_IDs:
            assert abs(query_rank.loc[match].values[0] - 0) < 1e-6


if __name__ == '__main__':
    unittest.main()
