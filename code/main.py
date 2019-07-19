import ast
import operator
import os
import pickle
from collections import Counter

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# Utility functions


# Tokenize sentence, remove stop words, lowercase words and remove punctuation.
def filter_tokenize(sentence):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(sentence.lower())

    filtered_tokens = [w for w in tokens if not w in stop_words and len(w) > 3]
    return filtered_tokens


# Lemmatize token through WordNet lemmatizer
def lemmatize(token):
    lmt = WordNetLemmatizer()
    return lmt.lemmatize(token)


# Stem token through English SnowballStemmer
def stem(token):
    stm = SnowballStemmer("english")
    return stm.stem(token)


# Apply tokenization, lemmatization and stemming on a document
def preprocess(doc):
    result = []
    for token in filter_tokenize(doc):
        tok = lemmatize(token)
        tok = stem(tok)
        result.append(tok)
    return result


# Dummy function to apply scikit-learn TfidfVectorizer on tokenized docs
def dummy_func(doc):
    return doc


# Term-frequency Inverse Document-Frequency provided by scikit-learn
# Input: List of tokenized docs
# Output: tf-idf model, Document x TF-IDF feature matrix, word features
def tfidf(docs):
    # build tf-idf model
    # exclude tokens that appear in less than 20% of the corpus
    # exclude tokens that appear in more than 40% of the corpus
    tf_idf = TfidfVectorizer(
        tokenizer=dummy_func,
        preprocessor=dummy_func,
        token_pattern=None,
        min_df=0.08,
        max_df=0.5
    )
    docs_index = docs.index.values

    docs = docs.values
    tf_idf.fit(docs)
    feature_names = tfidf.get_feature_names()

    tfs = tf_idf.transform(docs)
    tfidf_matrix = pd.DataFrame(tfs.todense(), index=docs_index, columns=feature_names)
    return tf_idf, tfidf_matrix, feature_names


# Execute pipeline to build TF-IDF model
def build(corpus):
    # Build the tokenized corpus, if not available locally
    if not os.path.isfile('content_tokens.csv'):
        processed_content = corpus['content'].map(preprocess)
        processed_content.to_csv('content_tokens.csv')
    else:
        processed_content = pd.read_csv('content_tokens.csv', index_col=0, header=None)
    processed_content = pd.DataFrame(processed_content.astype(str))
    docs = processed_content.iloc[:, 0].apply(ast.literal_eval)

    # Build set
    model, tf_idf, token_list = tfidf(docs)
    joblib.dump(tf_idf, 'tfidf.joblib')
    return tf_idf


# Iterative over each document's tokens
# Check if it appears in the query terms
# Increment document rank based on tf-idf for this document and this query term
def rank_documents(tf_idf, prep_query):
    rank = {'Ranking': np.zeros((len(tf_idf),))}
    query_rank = pd.DataFrame(rank, index=tf_idf.index.values)

    for doc in tf_idf.index.values:
        # print('=====Searching in doc ', doc)
        for word in tf_idf.loc[doc].index.values:
            # print('==Checking word ', word)
            if word in prep_query:
                # print(query_rank.loc[doc].values[0])
                query_rank.loc[doc] = query_rank.loc[doc] + tf_idf.loc[doc, word]
    query_rank.sort_values(by=['Ranking'], ascending=False, inplace=True)
    #total_score = query_rank['Ranking'].sum()
    #print('Total score: ', total_score)
    return query_rank


# Function that outputs the top 20 most relevant news articles, given a query
# Input: query
# Output: 20 most relevant articles, sorted by ranking score
def search_query(tf_idf, query, threshold):
    prep_query = preprocess(query)

    print('Searching for query: ', query)
    print('Preprocessed query: ', prep_query)

    query_rank = rank_documents(tf_idf, prep_query)
    # Total number of matched articles == positive tfidf scores
    matches = query_rank.loc[query_rank.Ranking > 0]
    total_n_matches = len(matches)

    print('\nTotal number of found matches: ', total_n_matches)
    # Top matches == highest tfidf scores
    top_match_IDs = []
    if not len(matches) == 0:
        top_match_IDs = matches.index.values[:threshold]
        print('\n===Top matched doc IDs===')
        print('Doc ID \t\t Ranking score')
        for match in top_match_IDs:
            print(match, '\t\t', query_rank.loc[match].values[0])
    return total_n_matches, query_rank, top_match_IDs


def run_query(tfidf_model):
    query = input("Input query: ")
    search_query(tfidf_model, query, 20)


def main():
    # load the dataset and drop unused columns (date year month url)
    # ID serves as index
    corpus = pd.read_csv('../dataset/articles1.csv', index_col=0, usecols=[1, 2, 3, 4, 9], header=0)

    if not os.path.isfile('tfidf.joblib'):
        tfidf_model = build(corpus)
    else:
        tfidf_model = joblib.load('tfidf.joblib')

    run_query(tfidf_model)


if __name__ == '__main__':
    main()
