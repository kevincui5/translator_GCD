# -*- coding: utf-8 -*-

#from pickle import load, HIGHEST_PROTOCOL
#from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
from trainer.util import *

# load dataset
raw_dataset = load_clean_sentences('english-german.csv')

n_sentences = len(raw_dataset)
# reduce dataset size for testing purpose
n_sentences = 1000
dataset = raw_dataset[:, :]
dataset = raw_dataset[:n_sentences, :]
# random shuffle (optional)
shuffle(dataset)
# split into train/test
ratio = 0.95
train_size = int(ratio*n_sentences)
train, test = dataset[:train_size], dataset[train_size:]
# save
#save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.csv')
save_clean_data(test, 'english-german-test.csv')