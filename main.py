import sys
import os.path
import torch
from torch.utils.data import DataLoader
import re
import torch.nn.functional as F
import torch.nn as nn

from vocabulary import Vocabulary
from word2vec import Word2Vec
from training import trainLoop
from trainingdataset import TrainingDataset

#files
wordlist = 'data/wordlist'
linelist = 'data/training_lines'
modelDataPath = 'data/model'

#settings
window = 3
features = 250
lr = 0.01
batchSize = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

#network initialization
vocabulary = Vocabulary()
vocabulary.load(wordlist)
print('Vocabulary size:', vocabulary.size)
trainingDataset = None

#load weights if saved before
w2v = None
if os.path.exists(modelDataPath): 
    print('Loading existing model...')
    w2v = torch.load(modelDataPath)
else:
    print('Initializing model...')
    w2v = Word2Vec(vocabulary.size, features).to(device)
    torch.save(w2v, modelDataPath)

while True:
    op = input("""1 - Evaluate vectors
2 - Initialize with new weights
3 - Start training
4 - View weights\n""")

    #evaluation
    if op == '1':
        first = input('First word: ')
        second = input('Second word: ')
        firstIndex = vocabulary.getIndex(first)
        secondIndex = vocabulary.getIndex(second)
        if firstIndex == None or secondIndex == None:
            print('Not in vocabulary')
        else:
            w2v.eval()
            print(w2v.similarity(firstIndex, secondIndex, vocabulary.size, device), '\n')

    #init neural network
    elif op == '2':
        confirm = input('Reset layers and save new weights? y/n: ')
        if confirm == 'y':
            w2v.initLayers()
            torch.save(w2v, modelDataPath)
            print("New weights initialized.\n")

    #training
    elif op == '3':
        loops = int(input("Loop: "))
        if trainingDataset == None:
            trainingDataset = TrainingDataset(linelist, vocabulary, window=window)
        w2v.train()
        for i in range(1, loops+1):
            print(f'Starting loop {i}.')
            trainLoop(w2v, vocabulary, trainingDataset, lr, batchSize, device, linelist, modelDataPath)
            torch.save(w2v, modelDataPath)
        print('Done.\n')

    #print weights
    elif op == '4':
        w2v.printWeights()
        print('\n')


