import numpy as np
import pandas as pd
import os
import csv
import pickle
import re
from nltk.corpus import stopwords, brown, words
from collections import defaultdict
from gensim import corpora, models, similarities
import sys

#READ FILE
#data[i] : the ith conversation
#data[i][j]: the jth token in the ith conversation
def read_file(file_path):
    stopWords = set(stopwords.words('english'))
    i = 0
    data = []
    for file_ in os.listdir(file_path):
        if i % 10000 == 0:
            print (file_)
        with open (file_path + file_) as infile:
            tmp = pd.read_table(infile, names = ['time', 'people1', 'people2', 'content'], quoting=csv.QUOTE_NONE)
            conversation = []
            for sentence in tmp['content']:
                if isinstance(sentence, float): continue
                for token in sentence.split():
                    token = re.sub(r'[~!@#$%^&\*\(\)_\+\-\=\[\]\{\}\\|,\.\/\`:;<>\?\'\"]+', '', token)
                    if token in stopWords or len(token) == 0: continue
                    conversation += token.lower(),
            data += conversation,
        i += 1
    return data

def filter_frequency(data, k):
    #get frequency of each token
    frequency = defaultdict(int)
    for conversation in data:
        for token in conversation:
            frequency[token] += 1
    return [[token for token in conversation if frequency[token] > k]
                for conversation in data]

def get_data_program():
    #read data
    file_path = 'data/dialogs/4/'
    data = read_file(file_path)
    #filter frequency
    flateen_texts = filter_frequency(data, 1)
    #create dictionary
    dictionary = corpora.Dictionary(flateen_texts)
    #save result
    dictionary.save('data/corpora.dict')
    with open('data/flateen_texts.pickle', 'wb') as handle:
        pickle.dump(flateen_texts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dictionary, flateen_texts

def train_model(texts, dictionary, K):
    #create model
    corpus = [dictionary.doc2bow(sentenct) for sentenct in texts]
    lda = models.LdaModel(corpus, num_topics=K)
    #get result
    topic_conversation = defaultdict(list)
    for sentence in texts:
        vec = lda[dictionary.doc2bow(sentence)]
        try:
            prob,topic = max([prob,topic] for topic, prob in vec)
            topic_conversation[topic].append(sentence)
        except:
            topic_conversation[K].append(sentence)
    return lda, topic_conversation


def train_model_program(K):
    with open('data/flateen_texts.pickle', 'rb') as handle:
        texts = pickle.load(handle)
    # train model
    dictionary = corpora.Dictionary.load('data/corpora.dict')
    lda, topic_conversation = train_model(texts, dictionary, K)

    #save results
    with open('data/topic_conversation.pickle', 'wb') as handle:
        pickle.dump(topic_conversation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/lda.pickle', 'wb') as handle:
        pickle.dump(lda, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return lda, topic_conversation

def create_topic():
    with open('data/topic_conversation.pickle', 'rb') as handle:
        topic_conversation = pickle.load(handle)

    # sort topic by popularity
    topic_rank = sorted([[topic, len(topic_conversation[topic])]for\
        topic in topic_conversation], key = lambda x : -x[1])

    #find key word which is not in brown dictionary
    wordSet = set(brown.words())
    topic_words_count = defaultdict(dict)
    total_words_count = defaultdict(int)
    for topic in topic_conversation:
        for sentence in topic_conversation[topic]:
            for word in sentence:
                if word not in wordSet:
                    topic_words_count[topic][word] =topic_words_count[topic].get(word, 0) + 1
                    total_words_count[word] += 1

    #caculate TFIDF score
    tfidf_score = defaultdict(dict)
    N = sum(total_words_count[word] for word in total_words_count)
    for topic in topic_words_count:
        for word in topic_words_count[topic]:
            tf = np.log(1 + topic_words_count[topic][word])
            idf = np.log(1 + N / total_words_count[word])
            tfidf_score[topic][word] = tf * idf
        tfidf_score[topic] = sorted(tfidf_score[topic].items(), key=lambda x : -x[1])

    #create map to map topicID to topic_name
    topic_map = {}
    for topic, count in topic_rank:
        topic_map[topic] = tfidf_score[topic][0][0]
    with open('data/topic_map.pickle', 'wb') as handle:
        pickle.dump(topic_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(K):
    #read data
    print ('K=', str(K))
    if not os.path.isfile('data/flateen_texts.pickle') or \
        not  os.path.isfile('data/corpora.dict'):
        get_data_program()
    if not os.path.isfile('data/topic_conversation.pickle') or \
        not os.path.isfile('data/lda.pickle'):
        train_model_program(K)
    create_topic()
    print ('train model and create topic done')



if __name__== "__main__":
  main(sys.argv[1])
