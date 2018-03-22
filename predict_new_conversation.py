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

def clear_conversation(file_path):
    conversation = []
    stopWords = set(stopwords.words('english'))
    with open (file_path) as infile:
        tmp = pd.read_table(infile, names = ['time', 'people1', 'people2', 'content'], quoting=csv.QUOTE_NONE)
        for sentence in tmp['content']:
            if isinstance(sentence, float): continue
            for token in sentence.split():
                token = re.sub(r'[~!@#$%^&\*\(\)_\+\-\=\[\]\{\}\\|,\.\/\`:;<>\?\'\"]+', '', token)
                if token in stopWords or len(token) == 0: continue
                conversation += token.lower(),
    return conversation

def find_topicID(file_path):
    conversation = clear_conversation(file_path)
    with open('data/lda.pickle', 'rb') as handle:
        lda = pickle.load(handle)
    dictionary = corpora.Dictionary.load('data/corpora.dict')

    vec = lda[dictionary.doc2bow(conversation)]
    try:
        prob,topic = max([prob,topic] for topic, prob in vec)
        return topic
    except:
        return None

def predict_topic(file_path):
    topic = find_topicID(file_path)
    with open('data/topic_map.pickle', 'rb') as handle:
        topic_map = pickle.load(handle)
    print ('topic ID= ' + str(topic))
    if not topic:
        return 'noise topic'
    else:
        return ('topic name= ' + topic_map[topic])

def main(file_path):
    #read data
    print (file_path)
    if not os.path.isfile('data/topic_conversation.pickle') or \
        not os.path.isfile('data/lda.pickle'):
        print ('you should run train_model.py first')
    else:
        print ('the topic of the conversation is:')
        print(predict_topic(file_path))

if __name__== "__main__":
  main(sys.argv[1])
