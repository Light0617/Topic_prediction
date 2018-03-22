import numpy as np
import pandas as pd
import os
import csv
import pickle
import re
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
from collections import defaultdict
from gensim import corpora

#READ FILE

def read_file(file_path):
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
                sen = []
                for token in sentence.split():
                    token = re.sub(r'[~!@#$%^&\*\(\)_\+\-\=\[\]\{\}\\|,\.\/\`:;<>\?\'\"]+', '', token)
                    if token in stopWords or len(token) == 0: continue
                    sen += token.lower(),
                conversation += sen,
            data += conversation,
        i += 1
    return data

file_path = 'data/dialogs/4/'
data = read_file(file_path)
# save data
# with open('data/data.pickle', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


frequency = defaultdict(int)
for conversation in data:
    for sentence in conversation:
        for token in sentence:
            frequency[token] += 1

texts = [[[token for token in sentence if frequency[token] > 1]
            for sentence in conversation]
            for conversation in data]

# with open('data/texts.pickle', 'wb') as handle:
#     pickle.dump(texts, handle, protocol=pickle.HIGHEST_PROTOCOL)

# flateen_texts1 = [[token for sentence in conversation for token in sentence] for conversation in texts]
#
# dictionary = corpora.Dictionary(flateen_texts1)
# dictionary.save('data/corpora.dict')
#
# with open('data/flateen_texts1.pickle', 'wb') as handle:
#     pickle.dump(flateen_texts1, handle, protocol=pickle.HIGHEST_PROTOCOL)
