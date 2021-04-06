from torch.utils.data import Dataset
import re
import math
import random

class TrainingDataset(Dataset):

    #store a list of (inputIndex, targetIndex) tuples
    data = []
    window = 5
    #subsampling
    corpusLength = 0
    wordCount = {}

    def __init__(self, path, vocabulary, window=5):
        self.window = window
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            #parse lines
            linecount = 0
            for line in lines:
                if linecount % 100000 == 0:
                    print(f"Lines parsed: {linecount}")
                tokens = list(filter(''.__ne__, re.split(r"[\ \t\n\.,:;*@#!?~_>\/-]+", line)))
                linecount += 1
                if len(tokens) > 1:
                    idxList = []
                    #get index for every word
                    for token in tokens:
                        tokenIndex = vocabulary.getIndex(token)
                        if tokenIndex != None:
                            idxList.append(tokenIndex)
                    #subsampling
                    for idx in idxList:
                        if idx in self.wordCount:
                            self.wordCount[idx] += 1
                        else:
                            self.wordCount[idx] = 1
                        self.corpusLength += 1

                        s = 0.001
                        perc = self.wordCount[idx] / self.corpusLength
                        prob = (math.sqrt(perc / s) + 1) * s / perc
                        if random.random() <= prob:
                            idxList.remove(idx)
                    if len(idxList) < 2:
                        continue
                    #get pairs and put them in self.data
                    for i in range(0, len(idxList)):
                        inputWord = idxList[i]
                        windowStart = i - self.window if i - self.window > 0 else 0
                        windowEnd = i + self.window + 1 if i + self.window + 1 < len(idxList) else len(idxList)
                        context = idxList[windowStart:i] + idxList[i+1:windowEnd]
                        for ctx in context:
                            self.data.append({'input':inputWord, 'target':ctx})


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]