import numpy as np
import pandas as pd
import os
import csv
import pickle
import re
from nltk.corpus import stopwords, brown, words
from collections import defaultdict
from gensim import corpora, models, similarities

def output_top_topic(K):
    with open('data/topic_conversation.pickle', 'rb') as handle:
        topic_conversation = pickle.load(handle)
    with open('data/topic_map.pickle', 'rb') as handle:
        topic_map = pickle.load(handle)
    topic_rank = sorted([[topic, len(topic_conversation[topic])]for\
        topic in topic_conversation], key = lambda x : -x[1])
    print ('the top ' + str(K) + ' topic is the following:')
    for topic, count in topic_rank[:K]:
        print(topic_map[topic])

def main():
    #read data
    if not os.path.isfile('data/topic_conversation.pickle') or \
        not os.path.isfile('data/lda.pickle'):
        print ('you should run train_model.py first')
    else:
        output_top_topic(10)

if __name__== "__main__":
  main()
