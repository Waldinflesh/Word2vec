from torch.utils.data import DataLoader
from trainingdataset import TrainingDataset
import torch
import torch.nn as nn
import datetime
import numpy

def trainLoop(w2v, vocabulary, trainingDataset, lr, batchSize, device, linelist, modelDataPath):
    w2v.train()
    print('Creating training dataset...')
    trainingDataLoader = DataLoader(trainingDataset, batch_size=batchSize, shuffle=True)
    print('Starting training...')

    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(w2v.parameters(), lr=lr)
    i = 0
    for samples in iter(trainingDataLoader):
        #prepare and forward vectors
        #target vector shape = batch size x 1
        #output vector shape = batch size x vocabulary size
        inputIndexList = samples['input'].numpy()
        sampleSize = len(inputIndexList)
        inputMatrix = torch.zeros(sampleSize, vocabulary.size, device=device)
        for j in range(0, sampleSize):
            inputMatrix[j, inputIndexList[j]] = 1

        #input and training vectors ready
        outputVector = w2v(inputMatrix)
        loss = lossFunc(outputVector, samples['target'].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 10_000 == 0):
            print(f'[{datetime.datetime.now()}] Loss at epoch {i}: {loss.item()}')
            torch.save(w2v, modelDataPath)
        i += 1