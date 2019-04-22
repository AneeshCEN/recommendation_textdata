from os import environ
import boto3
from flask import Flask, jsonify, Response
import json
import requests
import bottle
from threading import Thread
import os
import common.common_file as job_config
from common.s3_constants import *
import pandas as pd
import pickle


import pandas as pd
import numpy as np
import numpy as np
np.random.seed(1)

from textwrap import wrap

#!python -m nltk.downloader stopwords
import nltk
from nltk.tokenize import word_tokenize
import string
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#nltk.download('wordnet')

#nltk.download('punkt')


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
# from nltk.tokenize import RegexpTokenizer
# from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
#import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import ast


#aws_s3 = s3_constants["s3"]
#aws_access_key_id = s3_constants["aws_key"]
#aws_secret_access_key = s3_constants["aws_secret"]
#bucket = s3_constants["bucket_name"]




app = Flask(__name__)


def check_dir_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)




def download_models():

    s3 = boto3.resource('s3',aws_access_key_id= aws_access_key_id,
    aws_secret_access_key= aws_secret_access_key)
    #s3.download_file(bucket+'/models/doc2vec_model','/content/models/doc2vec_model')
    s3.Bucket('inputdatatraining').download_file('lsa_embeddings.pkl','content/models/lsa_embeddings.pkl')
    s3.Bucket('inputdatatraining').download_file('doc2vec_model','content/models/doc2vec_model')
    s3.Bucket('inputdatatraining').download_file('doctovec_embeddings.pkl','content/models/doctovec_embeddings.pkl')
    s3.Bucket('inputdatatraining').download_file('CurrentUrls.txt','content/models/CurrentUrls.txt')
    s3.Bucket('inputdatatraining').download_file('tfidf_model.pkl','/content/models/tfidf_model.pkl')
    s3.Bucket('inputdatatraining').download_file('svd_model.pkl','/content/models/svd_model.pkl')



class ProvideRecommendations():

    def __init__(self):
        check_dir_exist(os.getcwd()+'/content/models/')
        #download_models()

        self.dv = Doc2Vec.load(os.getcwd()+"/content/models/doc2vec_model")
        self.tf = pickle.load(open(os.getcwd()+"/content/models/tfidf_model.pkl", "rb"))
        self.svd = pickle.load(open(os.getcwd()+"/content/models/svd_model.pkl", "rb"))
        self.svd_feature_matrix = pickle.load(open(os.getcwd()+"/content/models/lsa_embeddings.pkl", "rb"))
        self.doctovec_feature_matrix = pickle.load(open(os.getcwd()+"/content/models/doctovec_embeddings.pkl", "rb"))

    def get_url(self, column):
        content_list = []
        content_url = open(os.getcwd()+'/content/models/CurrentUrls.txt', encoding='utf8')
        content_list.append(('').join(content_url.readlines()))
        lists = ast.literal_eval(content_list[0])
        for ind,val in enumerate(lists):
            if str(val[0]) in column:
                return lists[ind][1]

    def lemmatize(self, text):
        text_list = word_tokenize(text)
        stemmed_words = [wordnet_lemmatizer.lemmatize(i) for i in text_list]
        text = " ".join(stemmed_words)
        return text

    def make_lower_case(self,text):
        return text.lower()

    def remove_stop_words(self, text):
        text_list = word_tokenize(text)
        text = [i for i in text_list if i not in stop_words]
        text = " ".join(text)
        return text

    def remove_punctuation(self, text):
       # tokenizer = RegexpTokenizer(r'\w+')
        text_list = word_tokenize(text)
        text_list = [i for i in text_list if i not in string.punctuation]
        text_list = [x for x in text_list if not (x.isdigit())]
        text = " ".join(text_list)
        return text

    def clean_message(self, message):
        message = self.make_lower_case(message)
        message = self.remove_stop_words(message)
        message = self.remove_punctuation(message)
        message = self.lemmatize(message)
        return message

    def get_message_tfidf_embedding_vector(self, message):
        message_array = self.tf.transform([message]).toarray()
        message_array = self.svd.transform(message_array)
        message_array = message_array[:,0:25].reshape(1, -1)
        return message_array

    def get_message_doctovec_embedding_vector(self, message):
        message_array = self.dv.infer_vector(doc_words=message.split(" "), epochs=200)
        message_array = message_array.reshape(1, -1)
        return message_array

    def get_similarity_scores(self, message_array, embeddings):
        cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings,
                                                           Y=message_array,
                                                           dense_output=True))
        cosine_sim_matrix.set_index(embeddings.index, inplace=True)
        cosine_sim_matrix.columns = ["cosine_similarity"]
        return cosine_sim_matrix


    def get_ensemble_similarity_scores(self, message):
        message = self.clean_message(message)
        bow_message_array = self.get_message_tfidf_embedding_vector(message)
        semantic_message_array = self.get_message_doctovec_embedding_vector(message)

        bow_similarity = self.get_similarity_scores(bow_message_array, self.svd_feature_matrix)
        semantic_similarity = self.get_similarity_scores(semantic_message_array, self.doctovec_feature_matrix)

        ensemble_similarity = pd.merge(semantic_similarity, bow_similarity, left_index=True, right_index=True)
        ensemble_similarity.columns = ["semantic_similarity", "bow_similarity"]
        ensemble_similarity['ensemble_similarity'] = (ensemble_similarity["semantic_similarity"] + ensemble_similarity["bow_similarity"])/2
        ensemble_similarity.sort_values(by="ensemble_similarity", ascending=False, inplace=True)
        return ensemble_similarity





    def query_similar_docs(self, message, n):

        #love_message, hate_message = self.get_message_sentiment(message)

        similar_perfumes = self.get_ensemble_similarity_scores(message)
        subset = similar_perfumes.head(n)
        subset.drop(columns=['semantic_similarity', 'bow_similarity'], inplace=True)
        subset['recommended_article_name'] = list(subset.index)
        subset['recommended_article_url'] = subset['recommended_article_name'].apply(self.get_url)

        #dissimilar_perfumes = self.get_dissimilarity_scores(hate_message)
        #dissimilar_perfumes = dissimilar_perfumes.query('dissimilarity > .3')
        #similar_perfumes = similar_perfumes.drop(dissimilar_perfumes.index)

        return subset





@bottle.post("/prediction")
def traindocumentmodel():
    res = {}
    content=bottle.request.json
    pr = ProvideRecommendations()
    text_input = content.get('user_input')
    number = content.get('count')
    if number is None:
        number = 10
    if text_input is not None and type(text_input) == str:
        recs = pr.query_similar_docs(text_input, number)

        res = recs.to_json(orient='records')
    else:
        res['respose'] = 'invalid input'
    # print (res)
    return bottle.HTTPResponse(res)

if __name__ == '__main__':
    bottle.run()
