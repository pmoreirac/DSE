import ast
import os
import pickle
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


# Term-frequency Inverse Document-Frequency
# Input: List of tokenized docs
# Output: tf-idf model, Document x TF-IDF feature matrix, word features
def tfidf(docs):
    # build tf-idf model
    # exclude tokens that appear in less than 20% of the corpus
    # exclude tokens that appear in more than 40% of the corpus
    tfidf = TfidfVectorizer(
        tokenizer=dummy_func,
        preprocessor=dummy_func,
        token_pattern=None,
        min_df=0.2,
        max_df=0.5
    )
    docs_index = docs.index.values

    docs = docs.values
    tfidf.fit(docs)
    feature_names = tfidf.get_feature_names()

    #print('Index: ', docs_index, '\nFeature names: ', feature_names, '\n\n')

    tfs = tfidf.transform(docs)
    tfidf_matrix = pd.DataFrame(tfs.todense(), index=docs_index, columns=feature_names)
    print(tfidf_matrix)
    return tfidf, tfidf_matrix, feature_names


# Latent Dirichlet Allocation
# Input: document tf-idf matrix
# Output: Fitted LDA
def lda_fit(doc_term_matrix):
    LDA = LatentDirichletAllocation(
        n_components=10,
        n_jobs=-1
    )
    LDA.fit(doc_term_matrix)
    return LDA


# Display the top 10 highest probability words for each topic computed in LDA
def get_top_10_topic_words(lda, feature_names):
    print('Top 10 topic words')
    for topic in range(len(lda.components_)):
        print('\nTopic ', topic, ': ')
        top_words = lda.components_[topic].argsort()[-10:]
        for i in top_words:
            print(feature_names[i], end=" ")


# Execute pipeline to build search engine
def build(corpus):
    # Build the tokenized corpus, if not available locally
    if not os.path.isfile('content_tokens.csv'):
        processed_content = corpus['content'].map(preprocess)
        processed_content.to_csv('content_tokens.csv')
    else:
        processed_content = pd.read_csv('content_tokens.csv', index_col=0, header=None)
    processed_content = pd.DataFrame(processed_content.astype(str))
    docs = processed_content.iloc[:, 0].apply(ast.literal_eval)

    # Build tf-idf term matrix
    tfidf_model, tfidf_matrix, feature_names = tfidf(docs)

    """ EXPERIMENTAL
    # Feed matrix to LDA and generate topics
    lda = lda_fit(tfidf_matrix)
    # Check each topic's top 10 highest probability words
    get_top_10_topic_words(lda, feature_names)
    """
    # TODO: Store models locally for future use
    pickle.dump(tfidf_model, open('tfidf.p', 'wb'))
    #pickle.dump(lda, open('lda.p', 'wb'))

    return tfidf_model
# Function that outputs the top 20 most relevant news articles, given a query
# Input: query
# Output: 20 most relevant articles, sorted by ranking score
def search_query(tfidf_model, query):
    query = query.split(' ')

    test_doc = tfidf_model.transform([query])
    print(test_doc.todense())


def main():
    # load the dataset and drop unused columns (date year month url)
    # ID serves as index
    corpus = pd.read_csv('../dataset/articles1.csv', index_col=0, usecols=[1,2,3,4,9], header=0)

    if not (os.path.isfile('tfidf.p') and os.path.isfile('lda.p')):
        #tfidf_model, lda_model = build(corpus)
        tfidf_model = build(corpus)
    else:
        tfidf_model = pickle.load(open('tfidf.p', 'rb'))
        #lda_model = pickle.load(open('lda.p', 'rb'))
    search_query(tfidf_model, 'United States of America')


if __name__ == '__main__':
    main()